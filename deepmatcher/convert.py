r"""Convert deepmatcher ``.pth`` checkpoints to pickle-free ``.safetensors`` files.

    deepmatcher-convert model.pth                     # -> model.safetensors, verified
    deepmatcher-convert a.pth b.pth --out-dir out/    # several at once
    deepmatcher-convert model.pth --attr-summarizer rnn

Verification reloads the written file and compares every weight, vocabulary
and metadata entry bit for bit against the checkpoint; exit status 1 if any
differ.
"""
import argparse
import json
import logging
import os
import sys
import time

from .serialization import convert


def main(argv=None):
    p = argparse.ArgumentParser(prog='deepmatcher-convert', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+', help='.pth files written by save_state()')
    p.add_argument('--out-dir', help='write outputs here (default: next to each input)')
    p.add_argument('--attr-summarizer', default='hybrid',
                   help="MatchingModel attr_summarizer the checkpoint was trained with")
    p.add_argument('--model-kwargs', default='{}',
                   help='extra MatchingModel kwargs as JSON, e.g. \'{"hidden_size": 300}\'')
    p.add_argument('--no-verify', action='store_true', help='skip the round-trip check')
    p.add_argument('--json', action='store_true', help='print results as JSON lines')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    kwargs = {'attr_summarizer': args.attr_summarizer, **json.loads(args.model_kwargs)}
    failed = 0
    for path in args.checkpoints:
        stem = os.path.splitext(os.path.basename(path))[0] + '.safetensors'
        out = os.path.join(args.out_dir, stem) if args.out_dir else None
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
        start = time.perf_counter()
        result = convert(path, out, model_kwargs=kwargs, verify=not args.no_verify)
        result['seconds'] = round(time.perf_counter() - start, 1)
        result['input'] = path
        failed += result['verified'] is False
        if args.json:
            print(json.dumps(result))
        else:
            status = {True: 'verified', False: 'MISMATCH', None: 'not verified'}[result['verified']]
            print(f"{path} -> {result['output']}  {result['source_bytes'] / 1e6:.0f} MB -> "
                  f"{result['output_bytes'] / 1e6:.0f} MB  {status}  ({result['seconds']}s)")
            for diff in result['differences'][:20]:
                print('   ', diff)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
