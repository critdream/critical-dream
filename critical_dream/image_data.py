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
            f.write(response.content)


def main(multi_instance_config: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    with multi_instance_config.open() as f:
        multi_instance_config = yaml.safe_load(f)

    for conf in multi_instance_config:
        download_images(conf["instance_name"], conf["instance_urls"], output_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("--multi_instance_data_config", type=str)
    args = parser.parse_args()

    main(Path(args.multi_instance_data_config), Path(args.output_path))
