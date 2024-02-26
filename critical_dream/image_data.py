from pathlib import Path
import requests


image_urls = {
    "fjord": [
        "https://static.wikia.nocookie.net/criticalrole/images/8/80/Fjord_Level_20_portrait_-_Hannah_Friederichs.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/3/36/Fjord_Level_17_portrait_-_Ari.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/5/50/Fjord_Winter_Headshot.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/9/90/Fjordportrait2019.png",
        "https://static.wikia.nocookie.net/criticalrole/images/2/24/Fjord.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/c/c3/Fjord_Official_byArianaOrner.png",
        "https://static.wikia.nocookie.net/criticalrole/images/6/61/Fjord_by_Ariana_Orner.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/f/ff/Fjord_Winter.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/2/2b/Fjord_Level_20_-_Hannah_Friederichs.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/8/82/Rammaru_Fjord.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/9/9b/Fjord_-_Stephanie_Brown.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/8/86/Fjord_underwater_-_Linda_Lith%C3%A9n.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/f/f4/Fjord_-_Linda_Lithen.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/e/ee/Fjord_with_Star_Razor_-_heartofpack.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/1/1d/Fjord_-_Ari.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/0/0b/Fjord_and_Melora.jpg",
        "https://static.wikia.nocookie.net/criticalrole/images/a/ab/Fjord_star_razor_battletailors.jpeg",
        "https://static.wikia.nocookie.net/criticalrole/images/4/4f/Fjord_with_the_Cloven_Crystal_-_Dan_Bittencourt.jpg",
    ]
}


def download_images(character, urls, output_path):
    print(f"downloading images for {character}")
    for url in urls:
        response = requests.get(url)
        filepath = output_path / character / url.split("/")[-1]
        filepath.parent.mkdir(exist_ok=True)
        with filepath.open("wb") as f:
            f.write(response.content)


def main(output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    for character, urls in image_urls.items():
        download_images(character, urls, output_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(Path(args.output_path))
