import os
import shutil
import subprocess
import tempfile
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from mnists import MNIST

GITHUB_USERNAME = "nadirean"
GITHUB_TOKEN = os.environ.get("GH_TOKEN")
OUTPUT_IMAGE_NAME = "mnist_commits.png"

def get_owned_repos():
    """Fetches the names of all repositories owned by the user."""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    repos = []
    url = "https://api.github.com/user/repos?per_page=100&affiliation=owner"
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"GitHub API query failed with status code: {response.status_code}\n{response.text}")
        repos.extend(repo["name"] for repo in response.json())
        url = response.links.get("next", {}).get("url")
    return repos


def count_commits_in_repo(repo_name):
    """Clones a repo (blob-less) and counts unique commits reachable from any ref."""
    clone_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--bare", clone_url, tmpdir],
            check=True, capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-list", "--all", "--count"],
            cwd=tmpdir, check=True, capture_output=True, text=True,
        )
        return int(result.stdout.strip())


def get_total_commits():
    """Counts all commits across all owned repos, all branches, all time."""
    total = 0
    for repo in get_owned_repos():
        count = count_commits_in_repo(repo)
        print(f"{repo}: {count} commits")
        total += count
    print(f"Total commits found: {total}")
    return total

def generate_commit_image(commit_count):
    """
    Generates an image displaying the commit count using MNIST digits.
    """
    print("Loading MNIST dataset...")
    mnist = MNIST()
    x_train = mnist.train_images()
    y_train = mnist.train_labels()

    commit_str = str(commit_count)
    digit_images = []

    # Load Roboto-Black font with adjustable size
    font_path = "Roboto-Black.ttf"
    title_font_size = 32  # You can adjust this value as needed
    target_digit_height = 80
    target_digit_width = 80
    margin_x = 32
    margin_y = 20
    title_text = "Total Commits:"
    title_margin = 20
    min_width = 500
    min_height = 100

    try:
        font = ImageFont.truetype(font_path, title_font_size)
    except Exception as e:
        print(f"Could not load Roboto-Black.ttf, using default font. Error: {e}")
        font = ImageFont.load_default()

    print(f"Generating images for commit count: {commit_str}")
    # Seed deterministically so the digits only change when the count changes.
    np.random.seed(commit_count)
    for digit in commit_str:
        digit = int(digit)
        indices = np.where(y_train == digit)[0]
        random_index = np.random.choice(indices)
        digit_image_array = x_train[random_index]
        digit_image = Image.fromarray(digit_image_array.astype('uint8'), 'L')
        # Upscale digit
        digit_image = digit_image.resize((target_digit_width, target_digit_height), resample=Image.NEAREST)
        digit_images.append(digit_image)

    digits_width = sum(img.width for img in digit_images)
    digits_height = target_digit_height
    total_width = max(digits_width + 2 * margin_x, min_width)

    # Calculate title size using textbbox
    dummy_img = Image.new('L', (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    try:
        bbox = draw.textbbox((0, 0), title_text, font=font)
        title_w = bbox[2] - bbox[0]
        title_h = bbox[3] - bbox[1]
    except AttributeError:
        title_w, title_h = font.getsize(title_text)

    total_height = max(title_h + title_margin + digits_height + 2 * margin_y, min_height)

    # Create new image with black background
    combined_image = Image.new('L', (total_width, total_height), color=0)
    draw = ImageDraw.Draw(combined_image)

    # Draw title centered in white
    title_x = (total_width - title_w) // 2
    title_y = margin_y
    draw.text((title_x, title_y), title_text, font=font, fill=255)

    # Paste digits below title
    x_offset = (total_width - digits_width) // 2  # center digits horizontally
    y_offset = margin_y + title_h + title_margin

    for img in digit_images:
        combined_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

    combined_image.save(OUTPUT_IMAGE_NAME)
    print(f"Successfully generated and saved image as {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        raise ValueError("GitHub token not found. Please set the GH_TOKEN environment variable.")
    total_commits = get_total_commits()
    generate_commit_image(total_commits)