import requests
from bs4 import BeautifulSoup
import os
import urllib

# Function to create a directory to save images
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to download images
def download_images(url, directory):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            img_name = img_url.split('/')[-1]
            img_path = os.path.join(directory, img_name)
            try:
                urllib.request.urlretrieve(img_url, img_path)
                print(f"Image '{img_name}' downloaded successfully!")
            except Exception as e:
                print(f"Failed to download image '{img_name}': {str(e)}")

def main():
    url = input("Enter the URL to scrape images from: ")
    directory = input("Enter the directory name to save images (default: 'images'): ") or 'images'
    
    create_directory(directory)
    download_images(url, directory)

if __name__ == "__main__":
    main()


