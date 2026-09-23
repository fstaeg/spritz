import argparse
import json



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse MadGraph reweight card")
    parser.add_argument("input", help="Path to input json file with reweight structure from dump_reweight.py")
    args = parser.parse_args()

    input__ = json.load(open(args.input))

    for name, info in input__.items():
        print(f' "{name}": ' + '{')
        print(f'    "func": lambda events: events.LHEReweightingWeight[:, {info["idx"]}],')
        print(f'    "save_events": True,')        
        print(" },")        
        print()        