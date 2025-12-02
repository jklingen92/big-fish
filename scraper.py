import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import json
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pathlib import Path

import time
import re
import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

DEFAULT_IMG = "data:image/svg+xml;base64,CiAgICAgIDxzdmcgd2lkdGg9IjUwMCIgaGVpZ2h0PSIzMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgICAgICAgPHJlY3Qgd2lkdGg9IjUwMCIgaGVpZ2h0PSIzMDAiIGZpbGw9IiNmM2Y0ZjYiLz4KICAgICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyNTAsIDE1MCkiPgogICAgICAgICAgPGNpcmNsZSBjeD0iMCIgY3k9Ii0yMCIgcj0iMzAiIGZpbGw9IiNkMWQ1ZGIiIHN0cm9rZT0iIzljYTNhZiIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgICAgICAgICA8Y2lyY2xlIGN4PSItMTAiIGN5PSItMjUiIHI9IjgiIGZpbGw9IiM5Y2EzYWYiLz4KICAgICAgICAgIDxwYXRoIGQ9Ik0gLTIwLC0xMCBMIC0xMCwwIEwgMTAsLTIwIEwgMjAsLTEwIiBzdHJva2U9IiM5Y2EzYWYiIHN0cm9rZS13aWR0aD0iMyIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CiAgICAgICAgPC9nPgogICAgICAgIDx0ZXh0IHg9IjI1MCIgeT0iMjAwIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPgogICAgICAgICAgTm8gSW1hZ2UgQXZhaWxhYmxlCiAgICAgICAgPC90ZXh0PgogICAgICA8L3N2Zz4KICAgIA=="

class IGFAImageScraper:

    BASE_URL = "https://igfa.org"
    LOGIN_URL = "https://igfa.org/wp-login.php"
    DETAIL_URL = "https://igfa.org/member-services/world-record/detail/{}"
    
    def __init__(self, output_dir="angler_images"):
        """
        Initialize the scraper
        
        Args:
            base_url: URL template with a placeholder for ID (e.g., "https://example.com/angler/{}")
            output_dir: Directory to save images and metadata
        """
        self.output_dir = output_dir
        self.driver = self.init_driver(headless=False)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    def init_driver(self, headless=False):
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1200,800")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    
    def login(self, ):
        """Perform login using Selenium and transfer cookies to requests.Session"""
        try:
            self.driver.get(self.LOGIN_URL)
            print("Browser opened. Please complete the login in the browser window.")
            input("After you finish logging in, press Enter here to continue...")

            # Small pause to ensure cookies are set
            time.sleep(1)

        finally:
            try:
                self.driver.get(self.BASE_URL)
                
            except Exception:
                pass
    
    def download_image(self, img_url, angler_id):
        """Download an image and save it"""
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            
            # Determine file extension
            ext = img_url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = 'jpg'
            
            filename = f"angler_{angler_id}.{ext}"
            filepath = os.path.join(self.output_dir, "images", filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded: {filename}")
            return filename
        except Exception as e:
            print(f"Error downloading image from {img_url}: {e}")
            return None
    
    def scrape_angler(self, record_id):
        """Scrape a single angler page"""
        
        url = self.DETAIL_URL.format(record_id)
        print(f"\nScraping: {url}")
        
        self.driver.get(url)
        time.sleep(2)  # Wait for page to load
        response = self.driver.page_source
        
        soup = BeautifulSoup(response, 'html.parser')
        
        # Handle image
        rows = soup.find('h2', class_='font-bold text-xl')
        if not rows:
            print(f"No data found for ID {record_id} - skipping")
            return
        
        for row in rows:
            if row.get_text(strip=True) != "Record Photo":
                continue

            img = row.find_next('img')
            if img:
                img_src = img.get('src')
                if img_src == DEFAULT_IMG:
                    print(f"No image available for ID {record_id} - skipping")
                    return
                
                # Initialize metadata
                metadata = {"id": record_id}
                # Check if it's a base64 data URL
                if img_src and img_src.startswith('data:image'):
                    match = re.match(r'data:image/(\w+);base64,(.+)', img_src)
                    if match:
                        image_format = match.group(1)
                        base64_data = match.group(2)
                        
                        # Skip SVG placeholders (they're "No Image Available" graphics)
                        if image_format.lower() == 'svg+xml' or image_format.lower() == 'svg':
                            print(f"No image available for ID {record_id} - skipping")
                            metadata['image_file'] = None
                            metadata['image_source'] = 'none'
                        else:
                            # Decode base64 to binary for actual images
                            try:
                                image_data = base64.b64decode(base64_data)
                                
                                # Use jpeg extension for jpeg/jpg formats
                                ext = 'jpg' if image_format.lower() in ['jpeg', 'jpg'] else image_format.lower()
                                filename = f"angler_{record_id}.{ext}"
                                filepath = os.path.join(self.output_dir, "images", filename)
                                
                                with open(filepath, 'wb') as f:
                                    f.write(image_data)
                                
                                print(f"Downloaded: {filename}")
                                metadata['image_file'] = filename
                                metadata['image_source'] = 'base64'
                            except Exception as e:
                                print(f"Error decoding base64 image: {e}")
                                metadata['image_file'] = None
                                metadata['image_source'] = 'error'
                
                # Handle regular URL
                elif img_src and (img_src.startswith('http://') or img_src.startswith('https://')):
                    img_url = urljoin(url, img_src)
                    filename = self.download_image(img_url, record_id)
                    if filename:
                        metadata['image_file'] = filename
                        metadata['image_source'] = 'url'
            break
        else:
            print(f"No image tag found for ID {record_id} - skipping")
            return 

        # Check for h1 tag - skip if not present
        h1 = soup.find('h1', class_='text-5xl')
        if not h1:
            print(f"No h1 tag found - skipping ID {record_id}")
            return 
        
        # Extract common and scientific names
        common_name = h1.contents[0].strip() if h1.contents else ""
        scientific_name_span = h1.find('span', class_='font-normal')
        scientific_name = scientific_name_span.get_text(strip=True).strip('()') if scientific_name_span else ""
        
        metadata['common_name'] = common_name
        metadata['scientific_name'] = scientific_name
        
        # Extract length and girth
        rows = soup.find_all('div', class_='w-full flex flex-row')
        
        for row in rows:
            divs = row.find_all('div', recursive=False)
            if len(divs) == 2:
                label = divs[0].get_text(strip=True)
                value = divs[1].get_text(strip=True)
                
                if 'Length of Fish' in label:
                    length_match = re.search(r'([\d.]+)\s*cm\s*\(([\d.]+)\s*in\)', value)
                    if length_match:
                        metadata['length_cm'] = length_match.group(1)
                        metadata['length_in'] = length_match.group(2)
                
                elif 'Girth of Fish' in label:
                    girth_match = re.search(r'([\d.]+)\s*cm\s*\(([\d.]+)\s*in\)', value)
                    if girth_match:
                        metadata['girth_cm'] = girth_match.group(1)
                        metadata['girth_in'] = girth_match.group(2)
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, "metadata", f"record_{record_id}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def scrape_range(self, start_id, end_id, delay=1):
        """
        Scrape a range of angler IDs
        
        Args:
            start_id: Starting ID number
            end_id: Ending ID number (inclusive)
            delay: Delay between requests in seconds
        """
        results = []
        
        for record_id in range(start_id, end_id + 1):
            result = self.scrape_angler(record_id)
            if result:
                results.append(result)
            
            # Be polite - add delay between requests
            # time.sleep(delay)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Scraped {len(results)} anglers")
        print(f"✓ Data saved to: {self.output_dir}")
        
        return results


# Usage example
if __name__ == "__main__":
    # Configure these values
    START_ID = 41530
    END_ID = 99999

    scraper = IGFAImageScraper(output_dir="/media/jake/1D86-49D5/angler_images")
    scraper.login()
    scraper.scrape_range(START_ID, END_ID, delay=1)