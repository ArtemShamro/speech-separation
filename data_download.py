#!/usr/bin/env python3
import zipfile
from pathlib import Path

import requests
import typer
from tqdm import tqdm

app = typer.Typer(help="Download and extract datasets from Yandex Disk.")


def get_file_info(public_link: str) -> dict:
    """Получить информацию о файле по публичной ссылке"""
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources"
    params = {"public_key": public_link}
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    return response.json()


def get_direct_download_link(public_link: str) -> str:
    """Получить прямую ссылку для скачивания"""
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": public_link}
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    href = response.json().get("href")
    if not href:
        typer.echo(
            "❌ Failed to get direct download link. Check that the link is public."
        )
        raise typer.Exit(code=1)
    return href


def download_file(url: str, dest_path: str):
    """Скачать файл с прогресс-баром и корректными заголовками"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0 Safari/537.36"
    }
    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code == 403:
            raise RuntimeError(
                "403 Forbidden — файл недоступен. " "Проверь, что ссылка публичная."
            )
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


@app.command()
def download(
    link: str = typer.Argument(
        ..., help="Public Yandex.Disk link (e.g. https://disk.yandex.ru/d/XXXX)"
    ),
    dest: Path = typer.Argument(..., help="Destination directory for the dataset"),
    keep_archive: bool = typer.Option(
        False, "--keep-archive", help="Keep archive after extraction"
    ),
):
    """Download and extract dataset from Yandex.Disk"""
    dest.mkdir(parents=True, exist_ok=True)

    typer.echo("🔍 Getting file info...")
    info = get_file_info(link)
    file_name = info.get("name", "dataset.zip")
    archive_path = dest / file_name

    typer.echo(f"📄 File name: {file_name}")

    typer.echo("🔗 Getting direct download link...")
    file_url = get_direct_download_link(link)

    typer.echo(f"⬇️  Downloading to {archive_path}...")
    download_file(file_url, archive_path)
    typer.echo("\n📦 Extracting archive...")

    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(dest)
        if not keep_archive:
            archive_path.unlink()
        typer.echo(f"✅ Done! Files saved to: {dest.resolve()}")
    except zipfile.BadZipFile:
        typer.echo("⚠️ File is not a ZIP archive. Download completed but not extracted.")


if __name__ == "__main__":
    app()
