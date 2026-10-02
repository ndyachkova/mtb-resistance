#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
from Bio import Seq, SeqIO
#from Bio.Alphabet import IUPAC #this is not used
#from Bio.Blast.Applications import NcbiblastxCommandline # this is not used
from Bio.Blast import NCBIXML
from Bio import pairwise2
#from Bio.SubsMat.MatrixInfo import blosum62 # this is not used
import numpy as np
import os
import subprocess
import warnings
from Bio import BiopythonWarning
from datetime import timedelta
import time

# v9_shifts_strand
warnings.simplefilter('ignore', BiopythonWarning)
ref_path = 'data/h37rv.fasta' # 'data/h37rv_rev.fasta'
cds_path = 'data/gb_protein_coding.csv' #'data/mtb_protein_coding_rev_domains.tsv'
nc_path = 'data/gb_noncoding.csv' #'data/mtb_rna_rev.tsv'


def first_start_from_left(seq, blast_start, exp_start, broken_start, verbose=False):
    #i= len(seq)
    i = blast_start
    '''best_right_start = -1
    while i + 3 < len(seq):
        # this would be the left-most start
        if (seq[i:i + 3] == 'ATG') | (seq[i:i + 3] == 'GTG') | (seq[i:i + 3] == 'TTG'): 
            #print('Best start to right is', i)
            best_right_start = i
            break
        i += 3
    i = blast_start'''
    while i >= 3:
        #print(i-3, seq[i - 3:i])
        if (seq[i - 3:i] == 'ATG') | (seq[i - 3:i] == 'GTG') | (seq[i - 3:i] == 'TTG'): 
            return i - 3
        # if it's a stop codon: go search to right
        if seq[i-3:i] in ['TAA', 'TAG', 'TGA']:
            if verbose:
                print('Found stop to left, searching to right')
            break
        # if start is not changed and in this frame
        if (not broken_start) and exp_start == i-3:
            if verbose:
                print('Start remains but is not a start codon:', seq[i-3:i])
            return i - 3
        else:
            i -= 3
    if verbose:
        print('No start to left, trying right')
    # if not start to left
    i = blast_start
    while i + 3 < len(seq):
        # this would be the left-most start
        if (seq[i:i + 3] == 'ATG') | (seq[i:i + 3] == 'GTG') | (seq[i:i + 3] == 'TTG'): 
            return i
        i += 3
    return -1
 

def get_codon(protein_pos, nuc_seq):
    return nuc_seq[protein_pos*3 - 3:protein_pos*3]

def analyze_alignment(alignment, ref_seq, alt_seq, exp_start, true_start, exp_end, true_end, frameshift_index, pos_mutation, sample='', gene='', verbose=False):
    protein_ref = alignment[0].seqA
    protein_alt = alignment[0].seqB
    start_match = exp_start == true_start
    end_match = exp_end == true_end
    mutation_list = []
    al_start = alignment[0].start
    al_end = alignment[0].end
    # edges of problem-less middle
    center_start = alignment[0].start
    center_end = alignment[0].end
    frameshift_index_tmp = frameshift_index.copy()
    for m in frameshift_index_tmp:
        # true_end - 2: to not count stop-codon
        if frameshift_index[m] < true_start or frameshift_index[m] > true_end - 2:
            frameshift_index.pop(m)
    if verbose:
        if frameshift_index:
            print('Frameshifts inside gene:')
            for m in frameshift_index:
                print(m, frameshift_index[m])
    #sum((np.array(list(frameshift_index.values())) >= true_start) & (np.array(list(frameshift_index.values())) < true_end))
    if  len(frameshift_index) > 1:
        print(sample, 'Warning:', len(frameshift_index), 'frameshifts for', gene)
    # indeces of frameshifts inside the protein
    if verbose:
        print('Alignment coordinates:', al_start, al_end, 'Length', len(protein_alt))
        print(protein_ref[al_start:al_end])
        print(protein_alt[al_start:al_end])
    
    # find codons and indeces of frameshifts
    if frameshift_index:
        if verbose:
            if frameshift_index:
                print('Frameshifts:')
        for c in frameshift_index:
            # index in alt sequence
            codon = (frameshift_index[c] - true_start) // 3
            # index in alignment from index in alt
            tmp_ind = 0
            true_ind = 0
            while true_ind <= codon:
                if len(protein_alt) == tmp_ind:
                    tmp_ind += 1
                    true_ind += 1
                    break
                if protein_alt[tmp_ind] != '-':
                    true_ind += 1
                tmp_ind += 1
            codon = tmp_ind - 1
            if verbose:
                print(c, frameshift_index[c])
                #print(frameshift_index[c] - true_start)
                print('Codon', (frameshift_index[c] - true_start) // 3, 'index', codon)
            # coord in nuc seq, index in alignment, index in alt_seq
            frameshift_index[c] = [frameshift_index[c], codon, (frameshift_index[c] - true_start) // 3]
    
    # sort by index in alignment
    f_keys = sorted(list(frameshift_index.keys()), key=lambda x: frameshift_index[x][1])
    
    # if there are frameshifts that compensate each other
    # then whatever is between them cannot be properly aligned
    # and should be considered a long indel many-for-many
    # while everything to the edges is processed normally
    frameshifts = []
    if len(frameshift_index) > 1:
        for i in range(len(f_keys)):
            sh = len(pos_mutation[f_keys[i]][0]) - len(pos_mutation[f_keys[i]][1])
            for j in range(i+1, len(f_keys)):
                if verbose:
                    print(f'Frameshifts {i+1}, {j+1}:', pos_mutation[f_keys[i]], pos_mutation[f_keys[j]])
                sh += len(pos_mutation[f_keys[j]][0]) - len(pos_mutation[f_keys[j]][1])
                # if two frameshift compensate each other: the frame returns to normal after the second one
                if sh % 3 == 0:
                    if verbose:
                        print(f'Frameshifts {i+1} and {j+1} compensate')
                    # mark them as compensated later/earlier
                    frameshift_index[f_keys[i]] = [frameshift_index[f_keys[i]][0], frameshift_index[f_keys[i]][1], frameshift_index[f_keys[i]][2], f_keys[j]]
                    frameshift_index[f_keys[j]] = [frameshift_index[f_keys[j]][0], frameshift_index[f_keys[j]][1], frameshift_index[f_keys[j]][2], f_keys[i]]
                    frameshifts += [(f_keys[i], f_keys[j])]
                    break

    # check if start or end are frameshifts
    start_shift = False
    end_shift = False
    shift_in_left = []
    shift_in_right = []
    if frameshift_index:
        if verbose:
            if frameshift_index:
                print('Frameshifts:')
        for c in frameshift_index:
            # dist is a magic number! sometimes after frameshift a few aa remain the same
            dist = min(50, (al_end - al_start) // 2)
            codon = frameshift_index[c][1]
            if verbose:
                print(c, frameshift_index[c])
                #print(frameshift_index[c] - true_start)
                print('Codon', (frameshift_index[c][0] - true_start) // 3, 'index', codon)
            if (codon >= al_start-1) and (codon - al_start  < dist):
                # check if it's compensated
                # if earlier: set new al_start, process the indel in left_feature
                # if later: check that it's not too close to start 
                # (that it breaks orf and not creates it)
                # set al_start to second frameshift, send it to left_feature
                # if shift is compensated
                if len(frameshift_index[c]) == 4:
                    if verbose:
                        print('Compensated frameshift in left_feature')
                    # second frameshift is inside left_feature
                    if frameshift_index[c][-1] < c:
                        # both to not worry about the order
                        shift_in_left += [(c, frameshift_index[c][-1]), (frameshift_index[c][-1], c)]
                        al_start = max(codon+1, al_start)
                    # later: second frameshift is inside alignment
                    else:
                        # adding abs to get position after difference caused by frameshift
                        codon2 = frameshift_index[frameshift_index[c][-1]][1] + abs(len(pos_mutation[frameshift_index[c][-1]][0]) - len(pos_mutation[frameshift_index[c][-1]][1])) // 3 + 1
                        # check it's not too close to al_end
                        if (codon2 <= al_end) and (al_end - codon2  > dist):
                            shift_in_left += [(c, frameshift_index[c][-1]), (frameshift_index[c][-1], c)]
                            al_start = max(codon2+1, al_start)
                        else:
                            if verbose:
                                al_start = max(codon+1, al_start)
                                print('ORF is caused by compensated frameshift')
                # uncompensated shift
                else:
                    al_start = max(codon+1, al_start)
                start_shift = True
                start_match = False
                if verbose:
                    print('Start mismatch is caused by frameshift')
            elif (codon <= al_end) and (al_end - codon  < dist):
                # if it's compensated
                if len(frameshift_index[c]) == 4:
                    if verbose:
                        print('Compensated frameshift in right_feature')
                    # later
                    if frameshift_index[c][-1] > c:
                        shift_in_right += [(c, frameshift_index[c][-1]), (frameshift_index[c][-1], c)]
                        al_end = min(codon, al_end)
                    # earlier
                    else:
                        codon2 = frameshift_index[frameshift_index[c][-1]][1] + abs(len(pos_mutation[frameshift_index[c][-1]][0]) - len(pos_mutation[frameshift_index[c][-1]][1])) // 3 + 1
                        # check it's not too close to al_start
                        if (codon2 > al_start) and (codon2 - al_start > dist):
                            shift_in_right += [(c, frameshift_index[c][-1]), (frameshift_index[c][-1], c)]
                            al_end = min(codon2, al_end)
                        else:
                            if verbose:
                                al_end = min(codon, al_end)
                                print('ORF is caused by compensated frameshift')
                # uncompensated shift
                else:
                    al_end = min(codon, al_end)
                end_shift = True
                end_match = False
                if verbose:
                    print('End mismatch is caused by frameshift')
        if verbose and (start_shift or end_shift):
            print('New alignment coordinates are', al_start, al_end)
            print(protein_ref[al_start:al_end])
            print(protein_alt[al_start:al_end])
            print()
    
    # edges of problem-less middle
    center_start = al_start
    center_end = al_end

    # mismatch has a negative score
    # so theory: for local alignment start and end-1 are always matches (there is a warning if it's not the case)    
    identity = 0
    if verbose:
        print(protein_ref)
        print(protein_alt)
    if len(protein_ref) != len(protein_alt):
        print(sample, f'Warning: {gene} alignments have different lengths')
    # alignments + coords to analyze
    al_list = []
    # list for left features
    left_fts = []
    # list for right features
    right_fts = []

    # start of edge events
    # left feature: starts don't match
    if not start_match:
        left_ref = protein_ref[:al_start+1].replace('-', '')
        left_alt = protein_alt[:al_start+1].replace('-', '')
        while left_ref[-1] == '-' or left_alt[-1] == '-':
            left_ref = left_ref[:-1]
            left_alt = left_alt[:-1]
        # if start mismatch was caused by frameshift
        if start_shift:
            if verbose:
                print('Left frameshift:')
                print(left_ref)
                print(left_alt)
            left_fts += [(left_ref, left_alt)]
        # if it's already a left_feature
        elif protein_ref[0] == '-' or protein_alt[0] == '-':
            # up to first match
            left_fts += [(left_ref, left_alt)]
        # if it's not a left_feature but should be
        else:
            # if the start was broken but there is still a start codon there
            # so the only difference now is the start codon
            if left_ref[1:] == left_alt[1:]:
                # if it's a different codon
                #if ref_seq[:3] != alt_seq[:3]:
                # if it's a different aa
                if left_ref[0] != left_alt[0]:
                    mutation_list.append([left_ref[0], left_alt[0], 'alternative_start', ref_seq[:3], alt_seq[:3]])
                # if it's the same codon we do nothing
                # but if you want to do something, do it here in an else
            # can anything other than a frameshift cause this else?
            # a long indel at the edge can be mistaken for a edge feature - this is for that
            # you can check if there is an indel there if you really want
            else:
                print(sample, f"Warning: {gene} start shouldn't match but does",  protein_ref[:al_start+1], protein_alt[:al_start+1])
                left_fts += [(left_ref, left_alt)]

    # right feature
    if not end_match:
        right_ref = protein_ref[al_end-1:]
        right_alt = protein_alt[al_end-1:]
        while right_ref[0] == '-' or right_alt[0] == '-':
            right_ref = right_ref[1:]
            right_alt = right_alt[1:]
        right_ref = right_ref.replace('-', '')
        right_alt = right_alt.replace('-', '')
        if verbose:
            print('Right feature')
            print(right_ref)
            print(right_alt)
        # if start mismatch was caused by frameshift
        if end_shift:
            right_fts += [(right_ref, right_alt)]
        # if it is a right feature as it should be
        elif protein_ref[-1] == '-' or protein_alt[-1] == '-':
            # including last match
            right_fts += [(right_ref, right_alt)]
        # if it should be a right feature but isn't
        else:
            # if the stop was broken but there is still a stop codon there
            if (len(right_ref) == 1) & (len(right_alt) == 1):
                if ref_seq[-3:] != alt_seq[-3:]:
                    mutation_list.append(['alternative_stop', ref_seq[-3:], alt_seq[-3:]])
            else:
                print(sample, f"Warning: {gene} end shouldn't match but does", protein_ref[al_end-1:], protein_alt[al_end-1:])
                right_fts += [(right_ref, right_alt)]

    
    # to realign start: if start should match but doesn't
    if start_match and (protein_ref[0] == '-' or protein_alt[0] == '-'):
        # new_st is not defined
        st = 0
        # if start should match but doesn't
        if verbose:
            print("start should match but doesn't", st, protein_ref[:al_start+1], protein_alt[:al_start+1])
        # realign all except the firt_feature (all if there is none): left_ft might be really an indel further ahead
        alignment_start = pairwise2.align.globalms(protein_ref[:al_end].replace('-', ''), protein_alt[:al_end].replace('-', ''), 4, -2, -7, -1)
        #if verbose:
        #    print(alignment_start)
        st_match = False
        for a in reversed(alignment_start):
            #print(a)
            if a.seqA[0] != '-' and a.seqB[0] != '-': # find an alignment that shows matching start
                new_end = min(a.seqA.find('-'), a.seqB.find('-'))
                if new_end == -1:
                    new_end = max(a.seqA.find('-'), a.seqB.find('-'))
                # find the end of first gap: that should be where the left_feature moved
                while new_end < al_end and (a.seqA[new_end] == '-' or a.seqB[new_end] == '-'):
                    new_end += 1
                al_list += [(a.seqA[:new_end + 1], a.seqB[: new_end + 1], 0, new_end+1)]
                # right edge of processed sequence
                center_start = new_end
                st_match = True
                if verbose:
                    print('Realigned start:')
                    print(a)
                # [(alignment_start[0].seqA, alignment_start[0].seqB, 0, st+1)]
                break
        # if there is no good alignment: make the whole thing an ins (many for many)
        if not st_match:
            print(sample, 'Warning:', f'No alignment makes {gene} start match', protein_ref[:st+1], protein_alt[:st+1])
            center_start = st
            # make it left_feature
            left_fts += [(protein_ref[:st+1].replace('-', ''), protein_alt[:st+1].replace('-', ''))]
    
    # to realign end: if end should match but doesn't
    if end_match and (protein_ref[-1] == '-' or protein_alt[-1] == '-'):
        new_end = al_end - 1
        # if end should match but doesn't
        # new_end is al_end - 1
        if protein_ref[-1] == '-' or protein_alt[-1] == '-':
            while protein_ref[new_end] != protein_alt[new_end]: # is probably always false for local
                new_end -= 1
        if verbose:
            print("end should match but doesn't", new_end+1, protein_ref[new_end:], protein_alt[new_end:])
        alignment_end = pairwise2.align.globalms(protein_ref.replace('-', ''), protein_alt.replace('-', ''), 4, -2, -7, -1)
        end_matched = False
        for a in reversed(alignment_end):
            #print(a)
            if a.seqA[-1] != '-' and a.seqB[-1] != '-': # find an alignment that shows matching ends
                new_start = max(a.seqA.rfind('-'), a.seqB.rfind('-'))
                # find the end of first gap: that should be where the left_feature moved
                while new_start > al_start and (a.seqA[new_start] == '-' or a.seqB[new_start] == '-'):
                    new_start -= 1
                al_list += [(a.seqA, a.seqB, new_start, len(protein_ref))]
                # left edge of processed sequence
                center_end = new_start
                end_matched = True
                if verbose:
                    print('Realigned end:')
                    print(a)
                # [(alignment_start[0].seqA, alignment_start[0].seqB, 0, st+1)]
                break
        # if there is no good alignment: make the whole thing a right feature
        if not end_matched:
            print(sample, 'Warning:', f'No alignment makes {gene} end match', protein_ref[new_end:], protein_alt[new_end:])
            center_end = new_end
            # make it left_feature
            right_fts += [(protein_ref[new_end:].replace('-', ''), protein_alt[new_end:].replace('-', ''))]
            
    # processing left_features
    for left_ref, left_alt in left_fts:
        if verbose:
            print('Left feature:')
            print(left_ref)
            print(left_alt)
            #print('Global alignment of left feature:')
            #print(pairwise2.align.globalms(left_ref, left_alt, 4, -2, -7, -1))
        # if there are a few (>2?) matches on the left: it's probably not a broken start
        # but a result of two compensating frameshifts
        # and you should mark everything else as a long indel
        
        # make sure you get them w/o gaps
        # remove the matches on the right (keep one)
        tmp_ind = -1
        if min(len(left_ref), len(left_alt)) > 1:
            while left_ref[tmp_ind] == left_alt[tmp_ind]:
                tmp_ind -= 1
                if -tmp_ind >= min(len(left_ref), len(left_alt)):
                    break
            tmp_ind += 1
        if tmp_ind < -1:
            left_ref = left_ref[:tmp_ind]
            left_alt = left_alt[:tmp_ind]
        if len(left_ref) == 1 and  len(left_alt) == 1:
            mutation_list.append([left_ref, left_alt, 'alternative_start'])
        elif len(left_ref) == 1:
            mutation_list.append([left_alt[:-1], 'left_extension'])
        elif len(left_alt) == 1:
            mutation_list.append([left_ref[:-1], 'left_clip'])
        elif len(left_ref) > len(left_alt):
            mutation_list.append([left_ref, left_alt, 'broken_start_clip'])
        elif len(left_ref) < len(left_alt):
            mutation_list.append([left_ref, left_alt, 'broken_start_extension'])
        else:
            mutation_list.append([left_ref, left_alt, 'broken_start'])
    

    # processing right_features
    for right_ref, right_alt in right_fts:
        if verbose:
            print('Right feature:')
            print(right_ref)
            print(right_alt)
            #print('Global alignment of right feature:')
            #print(pairwise2.align.globalms(right_ref, right_alt, 4, -2, -7, -1))
        # if there are a few (>2?) matches on the right: it's probably not a broken end
        # but a result of two compensating frameshifts
        # and you should mark everything else as a long indel
        #
        # remove the matches on the left (leave one)
        tmp_ind = 0
        while right_ref[tmp_ind] == right_alt[tmp_ind]:
            tmp_ind += 1
            if tmp_ind >= min(len(right_ref), len(right_alt)):
                break
        if tmp_ind:
            right_ref = right_ref[tmp_ind-1:]
            right_alt = right_alt[tmp_ind-1:]
        if len(right_ref) == 1:
            mutation_list.append([right_alt[1:], 'right_extension'])
        elif len(right_alt) == 1:
            mutation_list.append([right_ref[1:], 'right_clip'])
        elif len(right_ref) > len(right_alt):
            mutation_list.append([right_ref, right_alt, 'broken_end_clip'])
        elif len(right_ref) < len(right_alt):
            mutation_list.append([right_ref, right_alt, 'broken_end_extension'])
        else:
            mutation_list.append([right_ref, right_alt, 'broken_end'])
        
    for f in frameshifts:
        if verbose:
            print('Frameshift pair:', f)
        if (f not in shift_in_left) and (f not in shift_in_right):
            codon1 = frameshift_index[f[0]][1]
            codon2 = frameshift_index[f[1]][1] + abs(len(pos_mutation[f[1]][0]) - len(pos_mutation[f[1]][1])) // 3 + 1
            if verbose:
                print('Codons:', codon1, codon2)
            if max(codon1, codon2) < al_start:
                if verbose:
                    print('Frameshift in left_feature')
            elif min(codon1, codon2) > al_end:
                if verbose:
                    print('Frameshift in right_feature')
            elif (min(codon1, codon2) < al_start) and (max(codon1, codon2) > al_start):
                print(sample, 'Warning: frameshift crosses alignment start in', gene)
            elif (min(codon1, codon2) < al_end) and (max(codon1, codon2) > al_end):
                print(sample, 'Warning: frameshift crosses alignment end in', gene)
            else:
                if verbose:
                    print('Before frameshift:', 0, codon1)
                    print(protein_ref[:codon1])
                    print(protein_alt[:codon1])
                # between left_feature and first frameshift
                al_list += [(protein_ref, protein_alt, center_start, codon1)]
                center_start = codon2
                if verbose:
                    print('Inside frameshift:', codon1, codon2)
                    print(protein_ref[codon1:codon2])
                    print(protein_alt[codon1:codon2])
                cur_ref = protein_ref[codon1:codon2].replace('-', '')
                cur_alt = protein_alt[codon1:codon2].replace('-', '')
                if verbose:
                    print('Frameshift feature:', codon1 - protein_ref[:codon1].count('-') + 1, cur_ref, cur_alt)
                if len(cur_ref) > len(cur_alt):
                    mutation_list.append([str(codon1 - protein_ref[:codon1].count('-') + 1), cur_ref, cur_alt, 'shift_del'])
                elif len(cur_ref) < len(cur_alt):
                    mutation_list.append([str(codon1 - protein_ref[:codon1].count('-') + 1), cur_ref, cur_alt, 'shift_ins'])
                else:
                    mutation_list.append([str(codon1 - protein_ref[:codon1].count('-') + 1), cur_ref, cur_alt, 'shift'])
                if verbose:
                    print('After frameshift:')
                    print(protein_ref[codon2:], codon2)
                    print(protein_alt[codon2:], codon2)

    # to skip the edge events
    # alignments + coords to analyze
    al_list += [(protein_ref, protein_alt, center_start, center_end)] # center w/o edge_feature problems
    
    for protein_ref, protein_alt, cur_start, cur_end in al_list:
        if verbose:
            print('Processing', cur_start, cur_end)
            print(protein_ref[cur_start:cur_end])
            print(protein_alt[cur_start:cur_end])
        gap_number_r = protein_ref[:cur_start].count('-')
        gap_number_a = protein_alt[:cur_start].count('-')
        # length of current indel
        indel_length = 0
        for i in range(cur_start, cur_end):
            if (protein_alt[i] == '-'):
                indel_length += 1
                gap_number_a += 1
                if indel_length == 1:
                    indel_start = i - 1
                mtype = 'del'
            elif (protein_ref[i] == '-'):
                gap_number_r += 1
                indel_length += 1
                if indel_length == 1:
                    indel_start = i - 1
                mtype = 'ins'
            else:
                pos = i - protein_ref[:i].count('-') + 1
                if indel_length > 0:
                    pos = indel_start - protein_ref[:indel_start].count('-') + 1
                    if mtype == 'ins':
                        if verbose:
                            print(str(pos), protein_ref[indel_start], protein_alt[indel_start:indel_start + 1 + indel_length], 'ins')
                        mutation_list.append([str(pos), protein_ref[indel_start], protein_alt[indel_start:indel_start + 1 + indel_length], 'ins']) 
                    #str(indel_start), mtype, indel_length))
                    # str(pos-1), protein_ref[m-1:m + mutation_dict[m][1]], protein_alt[m-1], 'del'
                    # str(pos), protein_ref[m], protein_alt[m:m + 1 + mutation_dict[m][1]], 'ins'
                    if mtype == 'del':
                        if pos != 0:
                            if verbose:
                                print(str(pos), protein_ref[indel_start:indel_start + 1 + indel_length], protein_alt[indel_start], 'del')
                            mutation_list.append([str(pos), protein_ref[indel_start:indel_start + indel_length + 1], protein_alt[indel_start], 'del'])
                        else:
                            print(sample, 'Warning: coordinate of del is 0 in', gene, protein_ref[indel_start-1:indel_start + indel_length], protein_alt[indel_start-1])
                    indel_length = 0
                alt_codon = get_codon(i - gap_number_a + 1, alt_seq)
                ref_codon = get_codon(i - gap_number_r + 1, ref_seq)
                # if animoacids are different
                # in case snp is right after the indel: otherwise it's coord in indel start
                pos = i - protein_ref[:i].count('-') + 1
                if protein_ref[i] != protein_alt[i]:
                    if pos == 1:
                        if verbose:
                            print(protein_ref[i], protein_alt[i], 'alternative_start', ref_codon, alt_codon)
                        mutation_list.append([protein_ref[i], protein_alt[i], 'alternative_start', ref_codon, alt_codon])
                    else:
                        if verbose:
                            print(str(pos), protein_ref[i], protein_alt[i], 'snp', ref_codon, alt_codon)
                        mutation_list.append([str(pos), protein_ref[i], protein_alt[i], 'snp', ref_codon, alt_codon])
                # if aminoacids are the same
                else:
                    identity += 1
                    if verbose and alt_codon != ref_codon:
                        if pos == 1:
                            print(protein_ref[i], protein_alt[i], 'alternative_start', ref_codon, alt_codon)
                        else:
                            print(str(pos), protein_ref[i], protein_alt[i], 'syn', ref_codon, alt_codon)
                    # the next part is to write syn mutations
                    '''
                    if alt_codon != ref_codon:
                        if pos == 1:
                            mutation_list.append([protein_ref[i], protein_alt[i], 'alternative_start', ref_codon, alt_codon])
                        else:
                            mutation_list.append([str(pos), protein_ref[i], protein_alt[i], 'syn', ref_codon, alt_codon])
                    '''
        # if indel_length > 0: # right feature
        if indel_length != 0:
            pos = indel_start - protein_ref[:indel_start].count('-') + 1
            if mtype == 'ins':
                if verbose:
                    print(str(pos), protein_ref[indel_start], protein_alt[indel_start:indel_start + 1 + indel_length], 'ins')
                mutation_list.append([str(pos), protein_ref[indel_start], protein_alt[indel_start:indel_start + 1 + indel_length], 'ins']) 
            #str(indel_start), mtype, indel_length))
            # str(pos-1), protein_ref[m-1:m + mutation_dict[m][1]], protein_alt[m-1], 'del'
            # str(pos), protein_ref[m], protein_alt[m:m + 1 + mutation_dict[m][1]], 'ins'
            if mtype == 'del':
                if pos != 1:
                    if verbose:
                        print(str(pos), protein_ref[indel_start:indel_start + 1 + indel_length], protein_alt[indel_start], 'del')
                    mutation_list.append([str(pos), protein_ref[indel_start:indel_start + indel_length + 1], protein_alt[indel_start], 'del'])
                else:
                    print(sample, 'Warning: coordinate of del is 0 in', gene, protein_ref[indel_start-1:indel_start + indel_length], protein_alt[indel_start-1])
    return identity/(al_end - al_start + 1), mutation_list 


# translate_and_align(subj, query, blast_start, blast_end, frame, ref_seq, break_threshold=threshold)
def translate_and_align(translated_old, query, blast_start, blast_end, frame, ref_seq, exp_start, exp_end, broken_start, frameshift_index, pos_mutation, sample='', gene='', break_threshold=0.7, verbose=False):
    if frame < 0:
        #myseq = Seq.reverse_complement(query) # already done
        true_start = first_start_from_left(myseq, len(query) - blast_end + 3, exp_start, broken_start, verbose=verbose)
    else:
        myseq = query
        true_start = first_start_from_left(myseq, blast_start + 2, exp_start, broken_start, verbose=verbose)
    if true_start != -1:
        translated_new = Seq.translate(myseq[true_start:], table='Bacterial', to_stop=True)
        new_length = len(translated_new) * 3 + 3
        true_end = true_start + new_length-1
        if verbose:
            print('Blast query coords:', blast_start, blast_end, 'frame', frame)
            print('True start is', true_start)
            print(myseq[true_start:true_start+22])
            #print(Seq.reverse_complement(myseq[true_start:true_start+22]))
            print('True end is', true_end)
        if translated_old == translated_new:
            if verbose:
                print('Translation is unchanged')
            return ['', translated_new, [], '', true_start, true_end]
        if (len(translated_new) <= len(translated_old) * break_threshold) | (len(translated_new) >= len(translated_old) * 1.3):
            if verbose:
                if len(translated_new) <= len(translated_old) * break_threshold:
                    if verbose:
                        print('Broken: too short', len(translated_new), 'instead of', len(translated_old))
                        print(myseq[true_start:true_start + 3 * len(translated_new) + 3])
                        print(Seq.translate(myseq[true_start:], table='Bacterial'))
                        '''if best_right_start != -1:
                            print('With right start:')
                            print(Seq.translate(myseq[best_right_start:], table='Bacterial'))
                            print(len(Seq.translate(myseq[best_right_start:], table='Bacterial', to_stop=True)))'''
                        print(translated_old[:len(translated_new)+1])
                else:
                    if verbose:
                        print("Broken: too long", len(translated_new), 'instead of', len(translated_old))
            return 'broken'
        else:
            # match mismatch gap_open gap_extension
            # if start and end remain the same, we expect no clips/extensions -> if there are ins close to the edges, global will keep them as ins
            # if there are frameshifts we do not trust edge matches
            no_frameshift = True
            for m in frameshift_index:
                if (frameshift_index[m] >= true_start) and (frameshift_index[m] <= true_end):
                    no_frameshift = False
                    break
            if true_start == exp_start and true_end == exp_end and no_frameshift:
                if verbose:
                    print('Global alignment')
                alignment = pairwise2.align.globalms(translated_old, translated_new, 4, -2, -7, -1)
                if verbose:
                    print(alignment[:3])
            # if start or end changed, we expect clips or extensions -> local prefers that
            else:
                if verbose:
                    print('Local alignment')
                alignment = pairwise2.align.localms(translated_old, translated_new, 4, -2, -7, -1)
                if alignment[0].seqA[alignment[0].start] != alignment[0].seqB[alignment[0].start]:
                    print(sample, f'Warning: {gene} local alignment start is not a match')
                if alignment[0].seqA[alignment[0].end-1] != alignment[0].seqB[alignment[0].end-1]:
                    print(sample, f'Warning: {gene} local alignment end is not a match')
            old_length = len(translated_old) * 3 + 3
            alt_seq = myseq[true_start:true_start + new_length]
            if frame < 0:
                ref_seq = Seq.reverse_complement(ref_seq)
            identity, mutation_list = analyze_alignment(alignment, ref_seq, alt_seq, exp_start, true_start, exp_end, true_end, frameshift_index, pos_mutation, sample=sample, gene=gene, verbose=verbose)
            if identity < break_threshold:
                if verbose:
                    print('Broken: identity', identity)
                return 'broken'
            return [alignment, translated_new, mutation_list, alt_seq, true_start, true_end] # true_start, true_end
    else: 
        if verbose:
            print("Broken: couldn't find start")
        return 'broken'
        
def get_mtype(ref, alt):
    if ref.isalpha() & (len(ref) == 1) & alt.isalpha() & (len(alt) == 1):
        return 'snp'
    elif (len(ref) < len(alt)) | (ref == '-'):
        return 'ins'
    else:
        return 'del'        
    

def main_work(input_f, nuc_ref, cds, nc_genes, blast_folder, outfolder='result/', domain_path='../domain_scores_new/', threshold=0.5, verbose=False):
    organism = input_f.split('/')[-1][:-9]
    nonref = pd.read_csv(input_f, sep='\t', header=None, names=['pos', 'ref', 'alt'], index_col=False)
    if not nonref.shape[0]:
        print('ERROR: no data avilable for', organism)
        return
    if domain_path:
        # hmmscan file
        out_file = f'{domain_path}{organism}_domains.txt'
        file = open(out_file, 'w')
    print('Starting', organism)
    start_time = time.perf_counter()
    outname = outfolder + organism + '_result.tsv'
    open(outname, 'w').close()
    
    #mutations dictionary
    positions = np.asarray(nonref.pos)
    pos_mutation = {}
    al = set('ATCG')
    for i in range(len(nonref)):
        mutation = nonref.iloc[i]
        pos = mutation['pos']
        alt = mutation['alt']
        ref = mutation['ref']
        if len(set(alt) - al) != 0:
            print('WARNING: illegal symbols in alt:', pos, ref, alt)
            continue
        pos_mutation[pos] = [ref, alt]
    # to make sure skipped mutations don't reappear later
    positions = np.array(list(pos_mutation.keys()))
    blast_temp = blast_folder + str(threshold) + organism
    
    flank = 180
    #get nonreference nucleotide sequence with flanking regions and reference protein sequence for each protein coding gene
    altered_positions = set()
    altered_broken = set()
    for i in range(len(cds)):
        seq = cds.iloc[i]
        #print(seq)
        start = seq['start']
        end = seq['end']
        output_info = ''
        inside_gene = list(positions[(positions >= start) & (positions <= end)])
        # if there are mutations inside gene
        if inside_gene:
            name = seq['name']
            strand = seq['strand']
            ref_seq = str(nuc_ref.seq[start-1:end])
            ref_seq_p = seq['translation']
            #if strand == '+':
            #    ref_seq_p = Seq.translate(ref_seq, table='Bacterial', to_stop=True)
            #else:
            #    ref_seq_p = Seq.translate(Seq.reverse_complement(ref_seq), table='Bacterial', to_stop=True)
            # true/false: do we check for domains in this gene
            domain = seq['domain']
            # for mutation indeces inside mutated string
            mut_index = {}
            frameshift_index = {}
            if verbose:
                print()
                print(name, start, end, strand)
                print('Mutations inside gene:', len(inside_gene))
                for c in inside_gene:
                    print(c, pos_mutation[c], end=' ')
                print()
                #print(changed)
                print(ref_seq[:7], '...', ref_seq[-7:], len(ref_seq))
                if strand == '-':
                    print('Reverse complement')
                    print(Seq.reverse_complement(ref_seq[-7:]), '...', Seq.reverse_complement(ref_seq[:7]), len(ref_seq))
            # check for frameshifts
            no_frameshift = True
            for c in inside_gene:
                if verbose:
                    print(pos_mutation[c])
                r, a = pos_mutation[c]
                # if it's a frameshift
                if abs(len(r) - len(a)) % 3 != 0:
                    no_frameshift = False
                    # to keep track of frameshifts
                    frameshift_index[c] = -1
            if verbose:
                print('No frameshift:', no_frameshift)
            # to allow maximum un-broken extension
            flank = int(0.3 * (end - start)+1) + 3
            changed = list(positions[(positions >= max(start - flank, 1)) & (positions <= min(end + flank, len(nuc_ref.seq)))])
            if verbose:
                print('Sequence edges:', max(start - flank, 1), min(end + flank, len(nuc_ref.seq)))
            changed.sort()
            if verbose:
                print('Changed:')
                for c in changed:
                    print(c, pos_mutation[c], end=' ')
                print()
            # index of ref start and end codons in alt sequence
            exp_start = -1
            exp_end = -1
            nonref_seq = str(nuc_ref.seq[max(start - flank, 0):changed[0] - 1]) + pos_mutation[changed[0]][1]
            len_mut = len(pos_mutation[changed[0]][0])
            mut_index[changed[0]] = len(nonref_seq) - len(pos_mutation[changed[0]][1])
            if exp_start == -1 and changed[0] >= start:
                #if verbose:
                #    print('Start is before first mutation')
                # len(nonref_seq) - len(pos_mutation[changed[0]][1]) is index of last mutation
                exp_start = len(nonref_seq) - len(pos_mutation[changed[0]][1]) + start - changed[0]
            if exp_end == -1 and changed[0] >= end:
                #if verbose:
                #    print('End is before first mutation')
                exp_end = len(nonref_seq) - len(pos_mutation[changed[0]][1]) + end - changed[0]
            for i in range(1, len(changed)):
                nonref_seq = nonref_seq + str(nuc_ref.seq[changed[i - 1] + len_mut - 1:changed[i] - 1]) + pos_mutation[changed[i]][1]
                len_mut = len(pos_mutation[changed[i]][0])
                # adding index in the string
                mut_index[changed[i]] = len(nonref_seq) - len(pos_mutation[changed[i]][1])
                # if we just passed the start
                if exp_start == -1 and changed[i] >= start:
                    #if verbose:
                    #    print(f'Start is before {i+1} mutation')
                    # len(nonref_seq) - len(pos_mutation[changed[0]][1]) is index of last mutation
                    exp_start = len(nonref_seq) - len(pos_mutation[changed[i]][1]) + start - changed[i]
                # if we just passed the end
                if exp_end == -1 and changed[i] >= end:
                    #if verbose:
                    #    print(f'End is before {i+1} mutation')
                    # len(nonref_seq) - len(pos_mutation[changed[0]][1]) is index of last mutation
                    exp_end = len(nonref_seq) - len(pos_mutation[changed[i]][1]) + end - changed[i]
            nonref_seq = nonref_seq + str(nuc_ref.seq[changed[-1] + len_mut - 1:min(end + flank, len(nuc_ref.seq))])
            if verbose:
                print('Mutations:')
                for c in mut_index:
                    print(c, pos_mutation[c], mut_index[c], nonref_seq[mut_index[c]])
            # if there are no mutations ater end
            if exp_end == -1:
                #if verbose:
                #    print('No mutations after end')
                exp_end = len(nonref_seq) - min(flank, len(nuc_ref.seq)-end) - 1
            if strand == '-':
                exp_start, exp_end = len(nonref_seq) - exp_end - 1, len(nonref_seq) - exp_start - 1
            if verbose:
                if strand == '+':
                    print('Expected start', exp_start, nonref_seq[exp_start:exp_start+22], Seq.translate(nonref_seq[exp_start:exp_start+22], table='Bacterial'))
                    print('Expected end', exp_end, nonref_seq[exp_end-22:exp_end+1])
                else:
                    print('Reverse complement:')
                    print('Expected start', exp_start, Seq.reverse_complement(nonref_seq)[exp_start:exp_start+22], Seq.translate(Seq.reverse_complement(nonref_seq)[exp_start:exp_start+22], table='Bacterial'))
                    print('Expected end', exp_end, Seq.reverse_complement(nonref_seq)[exp_end-22:exp_end+1])
            # invert everything for reverse complement
            if strand == '-':
                nonref_seq = Seq.reverse_complement(nonref_seq)
                ref_seq = Seq.reverse_complement(ref_seq)
                for mt in mut_index:
                    mut_index[mt] = len(nonref_seq) - mut_index[mt] - 1
            # indeces of frameshifts
            for c in frameshift_index:
                frameshift_index[c] = mut_index[c]
            if verbose:
                print('Mutations:')
                for c in mut_index:
                    print(c, pos_mutation[c], mut_index[c], nonref_seq[mut_index[c]])
                if frameshift_index:
                    print('Frameshifts:')
                    for c in frameshift_index:
                        print(c, pos_mutation[c], mut_index[c], nonref_seq[mut_index[c]])
            # expected sequence
            alt_seq = Seq.translate(nonref_seq[exp_start:exp_end+1], table='Bacterial', to_stop=True)
            # check if start and/or end are broken - we only really need this for no frameshift genes
            snp_broken_start = False
            snp_broken_end = False
            early_stop = False
            if no_frameshift:
                # here we process no_frameshift genes
                # so that functions for blast work with this
                if verbose:
                    print('No frameshift')
                frame = 1
                blast_start = exp_start+1
                blast_end = exp_end+1
                subj = ref_seq_p
                query = nonref_seq
                # broken start
                if nonref_seq[exp_start:exp_start+3] != ref_seq[:3]:
                    start_cod = nonref_seq[exp_start:exp_start+3]
                    if (start_cod != 'ATG') & (start_cod != 'GTG') & (start_cod != 'TTG'): 
                        snp_broken_start = True
                        if verbose:
                            print('broken start:', nonref_seq[exp_start:exp_start+3], 'instead of', ref_seq[:3])
                    else:
                        print(organism, 'Warning: alternative start in', name, nonref_seq[exp_start:exp_start+3], 'instead of', ref_seq[:3])
                # broken end
                if nonref_seq[exp_end-2:exp_end+1] != ref_seq[-3:]:
                    if nonref_seq[exp_end-2:exp_end+1] not in ['TAG', 'TAA', 'TGA']:
                        snp_broken_end = True
                        if verbose:
                            print('broken stop:', nonref_seq[exp_end-2:exp_end+1], 'instead of', ref_seq[-3:])
                    else:
                        print(organism, 'Warning: alternative stop in', name, nonref_seq[exp_end-2:exp_end+1], 'instead of', ref_seq[-3:])
                # EDIT after: to process no frameshift + early stop
                if len(alt_seq) != (exp_end - exp_start + 1) // 3 - 1:
                    early_stop = True
                    if verbose:
                        print('broken stop:', len(alt_seq), 'instead of', (exp_end - exp_start + 1) // 3 - 1)
                    # if early stop is too early for gene to remain unbroken: move the start to that early stop in case that fixes it (might not)
                    if len(alt_seq) < (exp_end - exp_start + 1) / 3 * threshold:
                        blast_start = exp_start + len(alt_seq) * 3 + 1
                        if verbose:
                            print('With this start gene is too short, changing to', blast_start)
                # we do not catch to early stops close together though: gene will be broken
                # EDIT after
            # if there are frameshifts - run blast
            else:
                if verbose:
                    print('Frameshift; running blast')
                # doing blastx (nonreference nucleotide vs reference protein sequence) and processing result
                with open(blast_temp + 'query.fasta', 'w') as temp:
                    temp.write(f'>{name}\n')
                    query = nonref_seq #nonref_dict[value][1]
                    temp.write(query)
                with open(blast_temp + 'subject.fasta', 'w') as subj_temp:
                    subj_temp.write('>subject\n')
                    subj = ref_seq_p #nonref_dict[value][2]
                    subj_temp.write(subj)
                #subprocess.run(['blastx','-out', blast_temp + 'res.xml', '-outfmt', '5', '-query', blast_temp + 'query.fasta', 
                #                '-evalue', '0.001', '-subject', blast_temp + 'subject.fasta', '-word_size', '2',
                #               '-query_gencode', '11'])
                '''if strand == '+':
                    d_strand = 'plus'
                else:
                    d_strand = 'minus'
                if verbose:
                    print('Strand:', d_strand)'''
                d_strand = 'plus' # minus is reverse complement by now
                # make database
                subprocess.run(['./diamond', 'makedb', '--in', blast_temp + 'subject.fasta', '-d', blast_temp + 'ref', 
                                    '--quiet'])
                # alignment
                subprocess.run(['./diamond', 'blastx',  '-d', blast_temp + 'ref', '-q', blast_temp + 'query.fasta', 
                                    '-o', blast_temp + 'res.xml', '--outfmt', '5', '--threads', '8',
                                    '--query-gencode', '11', '--evalue', '0.001', '--quiet', '--window', '2',
                                    '--strand', d_strand])
                with open(blast_temp + "res.xml") as result_handle:
                    try:
                        blast_record = NCBIXML.read(result_handle)
                    except:
                        # save altered mutations in broken
                        altered_broken.update(inside_gene)
                        with open(outname, 'a') as broken:
                            broken.write(name + '\tbroken_gene\n')
                        if verbose:
                            print('Broken: no good alignment')
                        print(organism, 'Warning: no good alignment for', name)
                        continue
                    else:
                        if not blast_record.alignments:
                            print(organism, 'Warning: no alignments in record for', name)
                            altered_broken.update(inside_gene)
                            with open(outname, 'a') as broken:
                                broken.write(name + '\tbroken_gene\n')
                            continue
                        best_al = blast_record.alignments[0]
                blast_start = best_al.hsps[0].query_start
                blast_end = best_al.hsps[0].query_end
                frame = best_al.hsps[0].frame[0]
            #ref_seq = nonref_dict[value][3]
            # if frameshits (ran blast) or broken edges or early stop: neew to find true start
            if (not no_frameshift) or snp_broken_start or snp_broken_end or early_stop:
                translation_res = translate_and_align(subj, query, blast_start, blast_end, frame, ref_seq, exp_start, exp_end, snp_broken_start, frameshift_index, pos_mutation, break_threshold=threshold, verbose=verbose, sample=organism, gene=name)
                if translation_res != 'broken':
                    if verbose:
                        print('Translated and aligned')
                    alignment, alt_seq, mutation_list, alt_nuc_seq, true_start, true_end = translation_res
                    # if the protein is unchanged
                    if not alignment:
                        altered_positions.update(inside_gene)
                        continue
            # if no frameshift and no problems
            else:
                #alt_seq = Seq.translate(nonref_seq[exp_start:exp_end+1], table='Bacterial', to_stop=True)
                # was done earlier
                true_end = exp_start+len(alt_seq)*3 + 2
                true_start = exp_start
                if verbose:
                    print('True start', true_start)
                    print('True end', true_end)
                if ref_seq_p == alt_seq:
                    if verbose:
                        print('Translation is unchanged')
                    altered_positions.update(inside_gene)
                    continue
                elif true_end == exp_end:
                    alignment = pairwise2.align.globalms(ref_seq_p, alt_seq, 4, -2, -7, -1)
                    if verbose:
                        print('Global alignment')
                        print(alignment)
                else:
                    if verbose:
                        print('Local alignment')
                    alignment = pairwise2.align.localms(ref_seq_p, alt_seq, 4, -2, -7, -1)
                    if alignment[0].seqA[alignment[0].start] != alignment[0].seqB[alignment[0].start]:
                        print(organism, f'Warning: {name} local alignment start is not a match')
                    if alignment[0].seqA[alignment[0].end-1] != alignment[0].seqB[alignment[0].end-1]:
                        print(organism, f'Warning: {name} locaalignment end is not a match')
                # if translations are not identical
                mutation_list = []
                translation_res = ''
                identity, mutation_list = analyze_alignment(alignment, ref_seq, nonref_seq[exp_start:exp_end+1], exp_start, true_start, exp_end, true_end, frameshift_index, pos_mutation, sample=organism, gene=name, verbose=verbose)
                if identity < threshold:
                    if verbose:
                        print('Broken: identity', identity)
                    translation_res = 'broken'
                else:
                    true_start = exp_start
                    #alt_nuc_seq = nonref_seq[exp_start:exp_end+1]
            #alt_pos = changed #nonref_dict[value][0]
            if translation_res == 'broken':
                # save altered mutation in broken
                altered_broken.update(inside_gene)
                with open(outname, 'a') as broken:
                    broken.write(name + '\tbroken_gene\n')
                continue
            mutations = len(inside_gene)
            features = 0
            mutation_list.sort(key=lambda x: int(x[0] if x[0].isdigit() else 0))
            with open(outname, 'a') as output:
                #analyze_mutations(value, changed, alignment, frame, pos_mutation, 
                #                  mutation_dict, output, alt_nuc_seq, ref_seq, altered_positions, true_start, true_end, exp_start, exp_end, verbose=verbose)
                if verbose:
                    print(len(mutation_list), 'mutations inside protein')
                '''for m in mutation_list:
                    if verbose:
                        print(m)
                    output.write('\t'.join([name] + m) + '\n') # saving mutations from pre-processed list
                '''
                # joining mutations on neighbouring positions: let's not do that
                '''cur_mut = mutation_list[0]
                for i in range(1, len(mutation_list)):
                    # if they are both positional mutations
                    if mutation_list[i][0].isdigit() and cur_mut[0].isdigit():
                        if int(mutation_list[i][0]) == int(cur_mut[0]) + 1:
                            if verbose:
                                print('Joining features:', mutation_list[i-1], mutation_list[i])
                            new_ref = cur_mut[1] + mutation_list[i][1]
                            new_alt = cur_mut[2] + mutation_list[i][2]
                            # if one of them is a shift we'll designate the whole thing a shift
                            if cur_mut[3].find('shift') != -1 or mutation_list[i][3].find('shift') != -1:
                                if len(new_ref) > len(new_alt):
                                    cur_mut = [cur_mut[0], new_ref, new_alt, 'shift_del']
                                elif len(new_ref) < len(new_alt):
                                    cur_mut = [cur_mut[0], new_ref, new_alt, 'shift_ins']
                                else:
                                    cur_mut = [cur_mut[0], new_ref, new_alt, 'shift']
                            else:
                                if len(new_ref) > len(new_alt):
                                    cur_mut = [cur_mut[0], new_ref, new_alt, 'del']
                                elif len(new_ref) > len(new_alt):
                                    cur_mut = [cur_mut[0], new_ref, new_alt, 'ins']
                                # what do we call this one
                                else:
                                    cur_mut = [cur_mut[0], new_ref, new_alt, 'indel']
                        else:
                            output.write('\t'.join([name] + cur_mut) + '\n')
                            cur_mut = mutation_list[i]
                            features += 1
                    else:
                        output.write('\t'.join([name] + cur_mut) + '\n')
                        cur_mut = mutation_list[i]
                        features += 1'''
                for mut in mutation_list:
                    output.write('\t'.join([name] + mut) + '\n')
                features += 1
            # marking mutations in extensions as processed
            in_left = 0
            in_right = 0
            if true_start != exp_start or true_end != exp_end:
                for c in mut_index:
                    if (mut_index[c] >= min(true_start, exp_start)) & (mut_index[c] <= max(true_start, exp_start)):
                        print(organism, c, f'is inside {name} left feature')
                        altered_positions.add(c)
                        in_left += 1
                    elif (mut_index[c] >= min(true_end, exp_end)) & (mut_index[c] <= max(true_end, exp_end)):
                        print(organism, c, f'is inside {name} right feature')
                        altered_positions.add(c)
                        in_right += 1
                    # if the mutation was to the left but it affected the sequence
                    elif (mut_index[c] >= min(true_start, exp_start)) and (mut_index[c] <= max(true_end, exp_end)) and c not in inside_gene:
                        print(organism, c, f'is inside {name}')
                        altered_positions.add(c)
                        mutations += 1
            mutations += max(0, in_left-1) + max(0, in_right - 1)
            altered_positions.update(inside_gene)
            if verbose:
                print(mutations, 'mutations,', features, 'features')
            if features > mutations:
                print(organism, 'Warning:', features, 'features out of', mutations, 'mutations in', name)
            # domain: if needed, domain-containing gene, non-syn mutations
            if domain_path and domain and len(mutation_list) != 0:
                # alt_seq is the protein sequence
                # save gene sequence to file
                fasta_file = blast_temp + 'query.fasta'
                res_file = blast_temp + 'res.txt'
                with open(fasta_file, 'w') as out:
                    print('>', name, sep='', file=out)
                    print(alt_seq, file=out)
                subprocess.run(['hmmscan', '--noali', '-o', '/dev/null', '--tblout', res_file, 
                            f'../data/hmm_profiles/{name}.hmm', fasta_file]) # hmmscan
                
                subprocess.run(['grep', '-v', '#', res_file], stdout=file, text=True)
    # hmm scores file
    if domain_path:
        file.close()
        os.remove(res_file)

    if os.path.exists(blast_temp + 'subject.fasta'):
        os.remove(blast_temp + 'subject.fasta')
        os.remove(blast_temp + 'query.fasta')
        os.remove(blast_temp + 'res.xml')
        os.remove(blast_temp + 'ref.dmnd')

    if verbose:
        print('Altered:', end=' ')
        for a in altered_positions:
            print(a, end=' ')
        print()
        if altered_positions:
            print()
        print('Broken:', end=' ')
        for a in altered_broken:
            print(a, end=' ')
        print()
            
    #### NON-CODING GENES
    #
    output = open(outname, 'a')
    if verbose:
        print('Non-coding')
    #non-coding genes annotated
    for g in range(len(nc_genes)): # edit 13.05.26: changed i to g
        seq = nc_genes.iloc[g]
        name = seq['name'] 
        start = seq['start']
        end = seq['end']
        strand = seq['strand']
        changed = list(positions[(positions >= start) & (positions <= end)])
        if changed:
            if verbose:
                print(name, start, end)
                ref_seq = nuc_ref.seq[start-1:end]
                if strand == '+':
                    print(ref_seq[:20])
                else:
                    print(Seq.reverse_complement(ref_seq)[:20])
                for c in changed:
                    print(c, pos_mutation[c], end=' ')
                print()
            for i in changed:
                # if this feature exists
                if i in altered_positions:
                    print(organism, name, 'overlaps with a gene at', i)
                    continue
                if i in altered_broken:
                    print(organism, name, 'overlaps with a broken gene at', i)
                    continue # normally this mutation is called by the name of the broken gene
                
                if strand == '+':
                    from_start = i - start + 1
                else:
                    from_start = end - i + 1
                mtype = get_mtype(pos_mutation[i][0], pos_mutation[i][1])
                if pos_mutation[i][0] == '-':
                    print('Reference is - for', name, 'at', from_start, i)
                if pos_mutation[i][1] == '-':
                    print('Alternative is - for', name, 'at', from_start, i)
                
                ref, alt = pos_mutation[i][0], pos_mutation[i][1]
                
                if strand == '-':
                    # need to ancor the indel to the other side
                    if mtype != 'snp':
                        coord_abs = i + len(ref)
                        from_start = from_start - len(ref)
                        # new ancor (on the other side)
                        ref_new = Seq.reverse_complement(nuc_ref[coord_abs-1])
                        if verbose:
                            print(Seq.reverse_complement(nuc_ref[coord_abs-3:coord_abs + 5]).seq)
                            print(Seq.reverse_complement(nuc_ref[coord_abs-1:coord_abs + 5]).seq)
                        # what used to be start is now at the end
                        # so we need to drop that and add new start to the begining
                        ref = ref_new + Seq.reverse_complement(ref)[:-1]
                        alt = ref_new + Seq.reverse_complement(alt)[:-1]
                    # if snp: reverse complement
                    else:
                        ref = Seq.reverse_complement(ref)
                        alt = Seq.reverse_complement(alt)
                output.write('\t'.join([name, str(from_start), ref, alt, mtype]) + '\n') # edit 04.05.26: remove rev_comp
            altered_positions.update(changed)
    output.close()

    if verbose:
        print('Gene promoters')
    # how far upstream is still promoter
    flank = 100
    for i in range(len(cds)):
        seq = cds.iloc[i]
        name = seq['name']
        start = seq['start']
        end = seq['end']
        strand = seq['strand']
        output_info = ''
        
        if strand == '+':
            prom = list(positions[(positions >= max(start - flank, 1)) & (positions < start)])
            prom.sort(reverse=True)
        else:
            prom = list(positions[(positions > end) & (positions <= min(end + flank, len(nuc_ref.seq)))])
            prom.sort()
        if prom:
            if verbose:
                print(name, start, end, strand, '(1-based)')
                for p in prom:
                    print(p, pos_mutation[p], end=' ')
                print()
            # mutation
            for p in prom:
                if verbose:
                    print(p)
                if p in altered_positions:
                    print(organism, name, 'promoter overlaps at', p)
                    continue
                if p in altered_broken:
                    print(organism, name, 'promoter overlaps with a broken gene at', p)
                    continue # normally this mutation is called by the name of the broken gene
                mtype = get_mtype(pos_mutation[p][0], pos_mutation[p][1])
                if strand == '+':
                    coord = f'{p - start}'
                else:
                    coord = f'{end - p}'
                if pos_mutation[p][0] == '-':
                    print('Reference is - for', name, 'at', coord, p)
                if pos_mutation[p][1] == '-':
                    print('Alternative is - for', name, 'at', coord, p)

                ref, alt = pos_mutation[p][0], pos_mutation[p][1]
                if strand == '-':
                    # need to ancor the indel to the other side
                    if mtype != 'snp':
                        coord_abs = p + len(ref)
                        coord = f'{int(coord) - len(ref)}'
                        # new ancor (on the other side)
                        try:
                            ref_new = Seq.reverse_complement(nuc_ref[coord_abs-1])
                        except:
                            print(coord_abs)
                            ref_new = Seq.reverse_complement(nuc_ref[coord_abs-1])
                        if verbose:
                            print(Seq.reverse_complement(nuc_ref[coord_abs-3:coord_abs + 5]).seq)
                            print(Seq.reverse_complement(nuc_ref[coord_abs-1:coord_abs + 5]).seq)
                        # what used to be start is now at the end
                        # so we need to drop that and add new start to the begining
                        ref = ref_new + Seq.reverse_complement(ref)[:-1]
                        alt = ref_new + Seq.reverse_complement(alt)[:-1]
                    # if snp: reverse complement
                    else:
                        ref = Seq.reverse_complement(ref)
                        alt = Seq.reverse_complement(alt)
                output_info += '\t'.join([name, coord, ref, alt, mtype]) + '\n' # edit 04.05.26: remove rev_comp
            altered_positions.update(prom)
            
        if output_info:
            with open(outname, 'a') as f:
                f.write(output_info)
        # end of edit E1.8

    if verbose:
        print('Non-coding promoters')
    # promoters for rnas
    for i in range(len(nc_genes)):
        seq = nc_genes.iloc[i]
        name = seq['name']
        start = seq['start']
        end = seq['end']
        strand = seq['strand']
        
        # edit E1.8
        # promoter
        output_info = ''
        if strand == '+':
            # NB start & end are 0-based, positions is 1-based
            prom = list(positions[(positions >= max(start - flank, 1)) & (positions < start)])
            prom.sort(reverse=True)
        else:
            prom = list(positions[(positions > end) & (positions <= min(end + flank, len(nuc_ref.seq)))])
            prom.sort()
        if prom:
            if verbose:
                print(name, start, end, strand, '(1-based)')
                for p in prom:
                    print(p, pos_mutation[p], end=' ')
                print()
            # mutation
            for p in prom:
                if verbose:
                    print(p)
                if p in altered_positions:
                    print(organism, name, 'promoter overlaps at', p)
                    continue
                if p in altered_broken:
                    print(organism, name, 'promoter overlaps with a broken gene at', p)
                    continue # normally this mutation is called by the name of the broken gene
                mtype = get_mtype(pos_mutation[p][0], pos_mutation[p][1])
                if strand == '+':
                    coord = f'{p - start}'
                else:
                    coord = f'{end - p}'
                if pos_mutation[p][0] == '-':
                    print('Reference is - for', name, 'at', coord, p)
                if pos_mutation[p][1] == '-':
                    print('Alternative is - for', name, 'at', coord, p)
                ref, alt = pos_mutation[p][0], pos_mutation[p][1]
                if strand == '-':
                    # need to ancor the indel to the other side
                    if mtype != 'snp':
                        coord_abs = p + len(ref)
                        coord = f'{int(coord) - len(ref)}'
                        # new ancor (on the other side)
                        ref_new = Seq.reverse_complement(nuc_ref[coord_abs-1])
                        if verbose:
                            print(Seq.reverse_complement(nuc_ref[coord_abs-3:coord_abs + 5]).seq)
                            print(Seq.reverse_complement(nuc_ref[coord_abs-1:coord_abs + 5]).seq)
                        # what used to be start is now at the end
                        # so we need to drop that and add new start to the begining
                        ref = ref_new + Seq.reverse_complement(ref)[:-1]
                        alt = ref_new + Seq.reverse_complement(alt)[:-1]
                    # if snp: reverse complement
                    else:
                        ref = Seq.reverse_complement(ref)
                        alt = Seq.reverse_complement(alt)

                output_info += '\t'.join([name, coord, ref, alt, mtype]) + '\n' # edit 04.05.26: remove rev_comp
            altered_positions.update(prom)
            
        if output_info:
            with open(outname, 'a') as f:
                f.write(output_info)

    if verbose:
        print("What's left")
    #all variants left
    for key in pos_mutation:
        if key not in altered_positions and key not in altered_broken:
            with open(outname, 'a') as output:
                if verbose:
                    print(key, pos_mutation[key])
                mtype = get_mtype(pos_mutation[key][0], pos_mutation[key][1])
                output.write('\t'.join(['-', str(key), pos_mutation[key][0], 
                                        pos_mutation[key][1], mtype]) + '\n')
    #print(input_f, 'ending')
    if verbose:
        print(outname)
    
    duration = timedelta(seconds=time.perf_counter()-start_time)
    print(organism, 'ending, total time', duration)


if __name__ == "__main__":
    #for file in files
    # run annot for one sample
    #file with variant calling result for an isolate (the list of variants)
    # make this path to samples
    input_path = sys.argv[1]
    # list of samples
    input_list = sys.argv[2]
    #output folder
    outfolder = sys.argv[3]
    #threshold for identity
    threshold = float(sys.argv[4])
    # first sample
    first_sample = int(sys.argv[5])
    # verbose
    verbose = (sys.argv[7] == 'T')
    # path to domain scores
    if len(sys.argv) == 9:
        domain_path = sys.argv[8]
    else:
        domain_path = ''
    #domain_path='../domain_scores_new/',
    # available sample files
    avail = os.listdir(input_path)
    # samples: presumed one filename per line
    with open(input_list) as f:
        smps = list([x.split()[0] for x in f.readlines()])
    # last sample
    if sys.argv[6] == '-':
        last_sample = len(smps)
    else:
        last_sample = int(sys.argv[6])
    # check that all requested samples are present in provided folder
    check = set(smps[first_sample:last_sample]) - set(avail)
    if len(check) != 0:
        print('Error: Files not in given directory:', check)
    # list of samples (paths to samples)
    samples = [os.path.join(input_path, x) for x in smps[first_sample:min(last_sample, len(smps))]]
    print(len(samples), 'samples')
    #reference sequence
    nuc_ref = SeqIO.read(ref_path, 'fasta')
    #MTB coding genes annotation
    #cds = pd.read_csv('data/mtb_protein_coding_rev.tsv', sep='\t')
    cds = pd.read_csv(cds_path, sep='\t')
    nc_genes = pd.read_csv(nc_path, sep='\t')
    #folder for blastx temporary files
    blast_folder = 'blastx_tempfiles/'
    os.makedirs(blast_folder, exist_ok=True)
    for sample in samples:
        main_work(sample, nuc_ref, cds, nc_genes, blast_folder, outfolder, domain_path, threshold, verbose)
    print('Finished with given samples')

