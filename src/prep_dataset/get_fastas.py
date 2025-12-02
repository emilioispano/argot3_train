#!/usr/bin/python3

import argparse
import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='This script gets FASTA sequences given UniProtKB accession numbers.')
    parser.add_argument('-p', '--prots', help='TXT file containing the protein IDs, one per line', required=True)
    parser.add_argument('-o', '--outfile', help='Output file', required=True)
    parser.add_argument('-t', '--threads', help='Number of threads', required=False, default=64, type=int)
    return vars(parser.parse_args())

def load_proteins(proteins_file):
    """Load protein IDs from a text file, one ID per line, and return them as a set."""
    with open(proteins_file, "r") as fp:
        proteins = {line.strip() for line in fp}
    return proteins

def download_fasta(protein):
    """Download the FASTA file for the given protein ID from UniProtKB."""
    url = f"https://www.uniprot.org/uniprot/{protein}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an error for bad responses (4xx, 5xx)
        return protein, response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {protein}: {e}", file=sys.stderr)
        return protein, None

def retrieve_fastas(proteins, out_path, filename, threads):
    """Download FASTA files for the given protein IDs and save them in the specified output path."""
    output_file = os.path.join(out_path, filename)
    with open(output_file, "w") as out_file, ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(download_fasta, protein): protein for protein in proteins}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading FASTA files"):
            protein, fasta = future.result()
            if fasta:
                out_file.write(fasta)

if __name__ == '__main__':
    args = get_args()
    prots = args['prots']
    outfile = args['outfile']
    cpus = args['threads']

    outpath = os.path.dirname(outfile)
    filename = os.path.basename(outfile)

    print(f"Loading proteins from {prots}...", file=sys.stderr)
    proteins = load_proteins(prots)

    print("Retrieving FASTA sequences...", file=sys.stderr)
    retrieve_fastas(proteins, outpath, filename, cpus)

    print("Done.", file=sys.stderr)
