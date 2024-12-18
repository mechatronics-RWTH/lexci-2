"""A tool for converting old Universal LExCI Master JSON config files to the new
YAML format.

File:   tools/universal_master_config_converter.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2024-11-22


Copyright 2024 Teaching and Research Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import sys
import argparse
import json
import ruamel.yaml
from typing import Any


def update(src_dict: dict[str, Any], dest_dict: dict[str, Any]) -> None:
    """Update the contents of `dest_dict` using `src_dict`. This method is
    needed because `ruamel.yaml`'s `update()` method doesn't preserve comments.

    Arguments:
        - src_dict: dict[str, Any]
              Dictionary to take the values from.
        - dest_dict: dict[str, Any]
              Dictionary to update.
    """

    for k, v in src_dict.items():
        if type(v) == dict:
            update(src_dict[k], dest_dict[k])
        else:
            dest_dict[k] = v


def main() -> None:
    """Main function of the tool."""

    # YAML loader/dumper that preserves comments
    yaml = ruamel.yaml.YAML(typ="rt")
    yaml.indent(mapping=4, sequence=4, offset=4)
    yaml.preserver_quotes = True

    # Parse command line arguments
    arg_parser = argparse.ArgumentParser(
        description=(
            "A tool for converting old Universal LExCI Master JSON config files"
            + " to the new YAML format."
        )
    )
    arg_parser.add_argument(
        "json_config_file", type=str, help="Config file in the old JSON format."
    )
    arg_parser.add_argument(
        "yaml_config_template",
        type=str,
        help=(
            "Template config file for the corresponding RL algorithm. This file"
            + " will be updated with the contents of the old config, i.e. it"
            + " will be overwritten!"
        ),
    )
    cli_args = arg_parser.parse_args(sys.argv[1:])

    # Load the old config file
    with open(cli_args.json_config_file, "r") as f:
        json_config = json.load(f)

    # Load the template
    with open(cli_args.yaml_config_template, "r") as f:
        yaml_config = yaml.load(f.read())

    # Update the template config and save it
    update(json_config, yaml_config)
    with open(cli_args.yaml_config_template, "w") as f:
        yaml.dump(yaml_config, f)


if __name__ == "__main__":
    main()
