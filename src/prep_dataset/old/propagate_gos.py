#!/usr/bin/python3

import argparse
import sys
from collections import defaultdict
import os
import pickle


def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-i', '--input', required=True)
	parser.add_argument('-c', '--clust_path', required=True)
	parser.add_argument('-g', '--gocentric', required=True)
	parser.add_argument('-o', '--output', required=True)

	return vars(parser.parse_args())


def parse_annots(infile):
	with open(infile, 'r') as fp:
		prots = set()
		for line in fp:
			prot, go = line.strip().split('\t')
			prots.add(prot)

	print(f'there are {len(prots)} prots with gos')
	return prots


def get_prot_sets(clustpath):
	prot_sets = defaultdict(set)
	gos = os.listdir(clustpath)
	print(f'Retrieving cd-hit high quality similarities...')
	len_gos = len(gos)
	for i, go in enumerate(gos):
		print(f'{go}: {i} / {len_gos}', end='\r', file=sys.stderr)
		db_90 = os.path.join(clustpath, os.path.join(go, 'db_90.clstr'))
		with open(db_90, 'r') as fp:
			for line in fp:
				if line.startswith('>'):
					continue
				elif line.startswith('0'):
					repr = line.split('|')[1]
					prot_sets[repr].add(repr)
				else:
					prot = line.split('|')[1]
					prot_sets[repr].add(prot)
					prot_sets[prot].add(repr)

	print(f'get_prot_sets retrieves a dictionary {len(prot_sets)} long')
	return prot_sets


def get_useful_prots(all_prots, prots_with_gos):
	alles = set()
	useful_prots = {}
	print(f'get_useful_prots takes in input lengths {len(all_prots)} and {len(prots_with_gos)}...')
	for prot in prots_with_gos:
		useful_prots[prot] = all_prots[prot]

	for _, prots in all_prots.items():
		for prot in prots:
			alles.add(prot)
	print(f'and returns useful_prots of len {len(useful_prots)}, and a set with len {len(alles)}')
	return useful_prots, alles


def parse_goa(prots, go_centric):
	print(f'Reading GO - centric GOA...')
	with open(go_centric, 'r') as fp:
		prots_gos = defaultdict(set)
		for i, line in enumerate(fp):
			if i % 1e6 == 0:
				print(f'line {i / 1e6} over 1004 c.ca', end='\r', file=sys.stderr)
			if line.startswith('>'):
				go = line.split('|')[0].strip('>')
			else:
				prot, mod, _ = line.split('\t')
				if prot in prots:
					if int(mod) < 2:
						prots_gos[prot].add(go.replace(':', '_'))

	print(f'parse_goa returns a dictionary {len(prots_gos)} long')
	return prots_gos


def link_prots_gos(prots_prots, prots_gos):
	annots = defaultdict(set)
	len_prots_prots = len(set(prots_prots.keys()))
	prots_with_gos = set(prots_gos.keys())
	print(f'link_prots_gos takes as input prots_prots of len {len(prots_prots)} and prots_gos of len {len(prots_gos)}...')
	for i, (repr, prots) in enumerate(prots_prots.items()):
		print(f'{repr}:::\t{i} / {len_prots_prots}', end='\r', file=sys.stderr)
		for go in prots_gos[repr]:
			annots[repr].add(go)
		for prot in prots.intersection(prots_with_gos):
			for go in prots_gos[prot]:
				annots[repr].add(go)

	print(f'...and returns a dictionary of len {len(annots)}')
	return annots


def write_results(prots, file):
	with open(file, 'w') as fp:
		len_prots = len(prots)
		for i, (prot, gos) in enumerate(prots.items()):
			print(f'{prot}:::\t{i} / {len_prots}', end='\r', file=sys.stderr)
			for go in gos:
				fp.write(f'{prot.strip()}\t{go.strip()}\n')


if __name__ == '__main__':
	args = get_args()
	gos_per_prots = args['input']
	clusters_path = args['clust_path']
	go_centric = args['gocentric']
	output_file = args['output']

	print(f'Parsing annotations from {gos_per_prots}...')
	prots_with_go = parse_annots(gos_per_prots)
	# gos_per_prots has the picked prots per keys and the gos it was picked by as values

	print(f'Reading cd-hit clusters from {clusters_path}...')
	prot_sets = get_prot_sets(clusters_path)
	# prot_sets has all proteins which share 90% identity with others and all those other proteins as values (if none, just itself)

	print(f'Discarding unwanted proteins...')
	prots_set, all_prots = get_useful_prots(prot_sets, prots_with_go)
	# prots_set has same keys as gos_per_prots and same values as prot_sets
	# all_prots is a simple set containing all prioteins of prot_sets.values()

	print(f'Reading annotations for all useful prots from {go_centric}...')
	gos_per_prots = parse_goa(all_prots, go_centric)
	# gos_per_prots now has all_prots as keys and high quality go terms read by go_centric_goa as values

	print(f'Propagating annotations for original proteins...')
	prots_gos = link_prots_gos(prots_set, gos_per_prots)
	# prots_gos has the same keys as prots_set and for values the intersection of all annotations prots_set.values() have in gos_per_prots.values()

	print(f'Writing results to {output_file}...')
	write_results(prots_gos, output_file)
