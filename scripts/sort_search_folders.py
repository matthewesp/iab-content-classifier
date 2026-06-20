#!/usr/bin/env python3
"""Sort _sort_data snowball-search video folders into the IAB taxonomy tree.

Each ``iab/_sort_data/snowball_search_*`` folder is a TikTok search whose videos
belong to exactly one node in the taxonomy. This script maps every search folder
to its taxonomy Unique ID, resolves the destination path from the taxonomy
itself (root -> leaf, by Name), and moves the .mp4 files there.

The mapping below accumulates across batches (Business & Finance, Careers,
Education, Crime, Disasters, ...). The script only acts on folders currently
present in _sort_data, so re-running after earlier batches were already sorted
is safe -- missing folders are skipped and existing destination files are not
overwritten.

Run with --dry-run first to preview. Without it, files are moved.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonomy import load_taxonomy

# search-folder name (under _sort_data/) -> taxonomy Unique ID.
# Keyed by Unique ID so destination paths come straight from the taxonomy and
# survive any Name punctuation (&, etc.).
FOLDER_TO_UID: dict[str, str] = {
    "snowball_search_Advertising_Industry": "91",
    "snowball_search_Agriculture_Industry": "102",
    "snowball_search_Angel_Investment": "64",
    "snowball_search_Apparel_Industry": "113",
    "snowball_search_Automotive_Industry": "117",
    "snowball_search_Aviation_Industry": "118",
    "snowball_search_Bankcruptcy": "65",                       # [sic] Bankruptcy
    "snowball_search_Biotech_and_Biomedical_Industry": "119",
    "snowball_search_Business_Accounting_and_Finance": "54",
    "snowball_search_Business_Aministration": "62",            # [sic] Administration
    "snowball_search_Business_Green_Solutions": "78",
    "snowball_search_Business_IT": "72",
    "snowball_search_Business_Loans": "66",
    "snowball_search_business_Logistics": "57",                # Business > Logistics
    "snowball_search_business_marketing_and_advertising": "58",
    "snowball_search_Business_Operations": "73",
    "snowball_search_business_sales": "59",
    "snowball_search_business_utilities": "79",
    "snowball_search_Civil_Engineering_Industry": "120",
    "snowball_search_Construction_Industry": "121",
    "snowball_search_Debt_Factoring___Invoice_Discounting": "67",
    "snowball_search_Defense_Industry": "122",
    "snowball_search_Economic_Commodities": "81",
    "snowball_search_Economic_Currencies": "82",
    "snowball_search_Economic_Financial_Crisis": "83",
    "snowball_search_Economic_Financial_Reform": "84",
    "snowball_search_Economic_Financial_Regulation": "85",
    "snowball_search_Economy_Gasoline_Prices": "86",
    "snowball_search_Economy_Housing_Prices": "87",            # -> Housing Market
    "snowball_search_Economy_Interest_Rates": "88",
    "snowball_search_Economy_Job_Market": "89",
    "snowball_search_Education_Industry": "92",
    "snowball_search_Entertainment_Industry": "93",
    "snowball_search_Environmental_Services_Industry": "94",
    "snowball_search_executive_leadership_and_management": "76",
    "snowball_search_Financial_Industry": "95",
    "snowball_search_Food_Industry": "96",
    "snowball_search_Government_Business": "77",
    "snowball_search_Healthcare_Industry": "97",
    "snowball_search_Hospitality_Industry": "98",
    "snowball_search_Human_Resources": "55",
    "snowball_search_Information_Services_Industry": "99",
    "snowball_search_Large_Businesses": "56",                  # -> Large Business
    "snowball_search_Legal_Services_Industry": "100",
    "snowball_search_Logistics_and_Transportation_Industry": "101",
    "snowball_search_Management_Consulting_Industry": "103",
    "snowball_search_Manufacturing_Industry": "104",
    "snowball_search_Mechanical_and_Industrial_Engineering_Industry": "105",
    "snowball_search_Media_Industry": "106",
    "snowball_search_Merger_and_acquisitions": "68",           # -> Mergers and Acquisitions
    "snowball_search_Metals_Industry": "107",
    "snowball_search_Non_Profit_Organizations_Industry": "108",
    "snowball_search_Pharmaceutical_Industry": "109",
    "snowball_search_Power_and_Energy_Industry": "110",
    "snowball_search_Private_Equity": "69",
    "snowball_search_Publishing_Industry": "111",
    "snowball_search_Real_Estate_Industry": "112",
    "snowball_search_recalls": "75",                           # Consumer Issues > Recalls
    "snowball_search_Retail_Industry": "114",
    "snowball_search_Sales___Lease_Back": "70",                # -> Sale & Lease Back
    "snowball_search_Small_and_Medium_sized_Businesses": "60", # -> Small and Medium-sized Business
    "snowball_search_Startups": "61",
    "snowball_search_Technology_Industry": "115",
    "snowball_search_Telecommunications_Industry": "116",
    "snowball_search_Venture_Capital": "71",

    # --- Careers (root 123) ---
    "snowball_search_Apprenticeships": "124",
    "snowball_search_Career_Advice": "125",
    "snowball_search_Career_Planning": "126",
    "snowball_search_Job_Fair": "128",                          # Job Search > Job Fairs
    "snowball_search_Resume_Writing_and_Advice": "129",         # Job Search > Resume Writing and Advice
    "snowball_search_Remote_Working": "130",
    "snowball_search_Vocational_Training": "131",

    # --- Education (root 132) ---
    "snowball_search_Adult_Education": "133",
    "snowball_search_College_Planning_": "138",                 # College Education > College Planning
    "snowball_search_Professional_Education_postgraduate": "139",  # College Education > Postgraduate Education
    "snowball_search_Undergraduate_Education": "141",           # College Education > Undergraduate Education
    "snowball_search_Early_Child_Education": "142",             # -> Early Childhood Education
    "snowball_search_Standardized_Testing": "144",             # Educational Assessment > Standardized Testing
    "snowball_search_Homeschooling": "145",
    "snowball_search_Homework_and_Study": "146",
    "snowball_search_Language_Learning": "147",
    "snowball_search_Online_Education": "148",
    "snowball_search_Primary_Education": "149",
    "snowball_search_Private_School": "134",
    "snowball_search_Secondary_Education": "135",
    "snowball_search_Special_Education": "136",

    # --- Crime / Disasters (top-level roots, no children) ---
    "snowball_search_Crime": "380",
    "snowball_search_Disasters": "381",

    # --- Events (root 8VZQHL) ---
    "snowball_search_Award_Shows": "162",                       # -> Awards Shows
    "snowball_search_Business_Expos_and_Confrences": "180",     # -> Business Expos & Conferences
    "snowball_search_Fan_Conventions": "185",

    # --- Family and Relationships (root 186) ---
    "snowball_search_Bereaverment": "187",                      # [sic] -> Bereavement
    "snowball_search_Dating": "188",
    "snowball_search_Divorce": "189",
    "snowball_search_Eldercare": "190",
    "snowball_search_Marriage_and_Civil_Unions": "191",
    "snowball_search_Single_Life": "200",
    "snowball_search_Adoption_and_Fostering": "193",            # Parenting > Adoption and Fostering
    "snowball_search_Daycare_and_preschool": "194",             # Parenting > Daycare and Pre-School
    "snowball_search_Interent_Safety_parenting": "195",         # [sic] Parenting > Internet Safety
    "snowball_search_Parenting_Babies_and_Toddlers": "196",
    "snowball_search_Parenting_Children": "197",                # -> Parenting Children Aged 4-11
    "snowball_search_Parenting_Teenagers": "198",               # -> Parenting Teens
    "snowball_search_Parenting_Special_Needs_Kids": "199",      # -> Special Needs Kids

    # --- Entertainment > Music (root 338) ---
    "snowball_search_Adult_Album_Alternative_music": "342",     # -> Adult Album Alternative
    "snowball_search_Adult_Contemporary_Music": "339",
    "snowball_search_soft_ac_music": "340",                     # Adult Contemporary > Soft AC Music
    "snowball_search_urban_ac_music": "341",                    # Adult Contemporary > Urban AC Music
    "snowball_search_Alternative_Music": "343",
    "snowball_search_Blues_Music": "360",                       # -> Blues
    "snowball_search_Childrens_Music": "344",                   # -> Children's Music
    "snowball_search_Classic_Hits": "345",
    "snowball_search_Classical_Music": "346",
    "snowball_search_College_Radio": "347",
    "snowball_search_Comedy_Music": "348",                      # -> Comedy (Music and Audio)
    "snowball_search_Contemporary_Hits_Pops_Top": "349",       # -> Contemporary Hits/Pop/Top 40
    "snowball_search_Country_music": "350",
    "snowball_search_Dance_and_Electronic_Music": "351",
    "snowball_search_Gospel_Music": "354",
    "snowball_search_Hiphop_Music": "355",                      # -> Hip Hop Music
    "snowball_search_New_Age_Music": "356",                     # -> Inspirational/New Age Music
    "snowball_search_Jazz_Music": "357",                        # -> Jazz
    "snowball_search_Oldies_Music": "358",                      # -> Oldies/Adult Standards
    "snowball_search_R_B_Soul_Funk_Music": "362",              # -> R&B/Soul/Funk
    "snowball_search_Reggae_Music": "359",                      # -> Reggae
    "snowball_search_Religious_Music_and_Audio": "361",         # -> Religious (Music and Audio)
    "snowball_search_Songwriters_Folk_Music": "353",           # -> Songwriters/Folk
    "snowball_search_Soundtracks_TV_and_Showtunes_music": "369",  # -> Soundtracks, TV and Showtunes
    "snowball_search_World_International_Music": "352",         # -> World/International Music
    "snowball_search_Urban_Contemporary_Music": "377",
    "snowball_search_Variety_Music": "378",                     # -> Variety (Music and Audio)
    # Rock Music (363) subtree
    "snowball_search_Album_Orient_Rock": "364",                 # -> Album-oriented Rock
    "snowball_search_Alternative_Rock": "365",
    "snowball_search_Classic_Rock": "366",
    "snowball_search_Hard_Rock": "367",
    "snowball_search_Soft_Rock": "368",

    # --- Entertainment (movies / television) ---
    "snowball_search_Movies": "324",
    "snowball_search_TV_Shows": "640",                          # -> Television

    # --- Fine Art (root 201) ---
    "snowball_search_Costumes": "202",                          # -> Costume
    "snowball_search_Dance": "203",
    "snowball_search_Design_Arts": "204",                       # -> Design
    "snowball_search_Digital_Arts": "205",
    "snowball_search_Fine_Art_Photography": "206",
    "snowball_search_Modern_Art": "207",
    "snowball_search_Opera": "208",
    "snowball_search_Theater_fine_arts": "209",                 # -> Theater

    # --- Food & Drink (root 210) ---
    "snowball_search_Alcoholic_Beverages": "211",
    "snowball_search_Vegan_Diets": "212",
    "snowball_search_Vegatarian_Diets": "213",                  # [sic] -> Vegetarian Diets
    "snowball_search_World_Cuisines": "214",
    "snowball_search_Barbecues_and_Grilling": "215",
    "snowball_search_Cooking": "216",
    "snowball_search_dessert_and_baking": "217",                # -> Desserts and Baking
    "snowball_search_Dining_out": "218",                        # -> Dining Out
    "snowball_search_Food_Allergies": "219",
    "snowball_search_healthy_cooking_and_eating": "221",        # -> Healthy Cooking and Eating
    "snowball_search_Non-Alcoholic_Beverages": "222",
}


def _sanitize(name: str) -> str:
    """Mirror build_taxonomy_folders.py so destinations match existing folders."""
    return name.replace("/", "-").replace("\\", "-").strip() or "_unnamed_"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iab", type=Path, default=Path("iab"),
                   help="root iab/ folder (default: ./iab)")
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--dry-run", action="store_true",
                   help="preview moves without touching files")
    p.add_argument("--remove-empty", action="store_true",
                   help="remove a search folder after its videos are moved")
    args = p.parse_args()

    tax = load_taxonomy(args.taxonomy)
    sort_root = args.iab / "_sort_data"

    # Validate mapping against what's actually on disk before moving anything.
    on_disk = {d.name for d in sort_root.iterdir() if d.is_dir()}
    mapped = set(FOLDER_TO_UID)
    if missing := (mapped - on_disk):
        print(f"WARNING: mapped folders not found on disk: {sorted(missing)}", file=sys.stderr)
    if unmapped := (on_disk - mapped):
        print(f"WARNING: search folders with no mapping (skipped): {sorted(unmapped)}", file=sys.stderr)

    total_moved = 0
    for folder, uid in sorted(FOLDER_TO_UID.items()):
        src = sort_root / folder
        if not src.is_dir():
            continue
        dest = args.iab.joinpath(*[_sanitize(tax.nodes[n].name) for n in tax.path_to(uid)])
        if not dest.is_dir():
            print(f"ERROR: destination missing for {folder} -> {dest}", file=sys.stderr)
            continue

        vids = sorted(src.glob("*.mp4"))
        for v in vids:
            target = dest / v.name
            if args.dry_run:
                pass
            else:
                if target.exists():
                    print(f"  skip (exists): {target}", file=sys.stderr)
                    continue
                shutil.move(str(v), str(target))
        total_moved += len(vids)
        print(f"{folder:>55s}  ->  {dest.relative_to(args.iab)}  ({len(vids)} videos)")

        if args.remove_empty and not args.dry_run:
            leftover = [f for f in src.iterdir() if f.name != ".DS_Store"]
            if not leftover:
                for f in src.iterdir():
                    f.unlink()
                src.rmdir()

    verb = "Would move" if args.dry_run else "Moved"
    print(f"\n{verb} {total_moved} videos across {len(FOLDER_TO_UID)} searches.", file=sys.stderr)


if __name__ == "__main__":
    main()
