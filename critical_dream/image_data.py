import shutil

from pathlib import Path
import requests
import yaml


def download_images(character, urls, output_path):
    print(f"downloading images for {character}")
    for url in urls:
        print(f"getting image from url: {url}")
        response = requests.get(url)
        filepath = output_path / character / url.split("/")[-1]
        filepath.parent.mkdir(exist_ok=True)
        with filepath.open("wb") as f:
            print(f"saving image to {filepath}")
            f.write(response.content)
            

def main(multi_instance_config: Path, output_path: Path, delete_existing: bool):
    output_path.mkdir(parents=True, exist_ok=True)
    with multi_instance_config.open() as f:
        multi_instance_config = yaml.safe_load(f)

    for conf in multi_instance_config:
        if delete_existing:
            p = output_path / conf["instance_name"]
            if p.exists():
                print(f"deleting existing directory: {p}")
                shutil.rmtree(p)
        download_images(
            conf["instance_name"],
            conf["instance_urls"],
            output_path,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("--multi_instance_data_config", type=str)
    parser.add_argument("--delete_existing", action="store_true")
    args = parser.parse_args()

    main(
        Path(args.multi_instance_data_config),
        Path(args.output_path),
        args.delete_existing,
    )
