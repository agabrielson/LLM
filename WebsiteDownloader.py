#!/usr/bin/env python3

import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque

class WebsiteDownloader:
    def __init__(self, output_dir="downloaded_sites"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_page(self, url, html):
        """Save HTML to file, sanitize filename."""
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        if not path:
            path = "index"
        filename = f"{parsed.netloc}_{path}.html"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved: {filepath}")

    def get_links(self, url, html):
        """Extract absolute links from a page."""
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute = urljoin(url, href)
            if urlparse(absolute).netloc == urlparse(url).netloc:
                links.add(absolute)
        return links

    def crawl(self, start_urls, max_depth=2):
        """Crawl starting from list of URLs up to max_depth."""
        visited = set()
        queue = deque([(url, 0) for url in start_urls])

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                html = response.text
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue

            self.save_page(url, html)
            visited.add(url)

            if depth < max_depth:
                links = self.get_links(url, html)
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))


if __name__ == "__main__":
    # List of websites to start from
    websites = [
        "https://en.wikipedia.org/wiki/List_of_firearm_brands",
        "https://gununiversity.com/gun-reviews/",
        "https://www.pewpewtactical.com/gun-gear-reviews/",
        "https://www.thetruthaboutguns.com/",
        "https://www.guns.com/news/category/reviews",
        "https://www.gunsandammo.com/",
        "https://en.wikipedia.org/wiki/List_of_individual_weapons_of_the_U.S._Armed_Forces",
        "https://gunmagwarehouse.com/blog/the-history-of-u-s-military-issue-sidearms-from-flintlocks-to-the-m17/",
        "https://www.pewpewtactical.com/police-sidearms-past-present/",
        "https://www.americanrifleman.org/content/by-the-decade-wiley-clapp-s-favorite-handgun-2010s/",
        "https://www.americanrifleman.org/content/wiley-clapp-s-five-favorite-firearms/",
        "https://www.gun-tests.com/handguns/colt-talo-single-action-army-wiley-clapp-review/",
        "https://www.peacemakerspecialists.com/gunsmithing/",
        "https://www.americanrifleman.org/content/collecting-second-generation-colt-single-action-army-revolvers/",
        "https://gundigest.com/gun-collecting/colt-saa-collecting",
        "https://en.wikipedia.org/wiki/Colt_Single_Action_Army",
        "https://gunmagwarehouse.com/blog/a-short-history-of-the-1911/",
        "https://www.turnbullrestoration.com/colt-1911-exploring-the-classics/",
        "https://en.wikipedia.org/wiki/M1911_pistol",
        "https://en.wikipedia.org/wiki/Category:Smith_%26_Wesson_revolvers",
        "https://gunprime.com/blog/best-2011-pistols-the-ultimate-guide",
        "https://www.pewpewtactical.com/best-revolvers/",
        "https://www.shootingtimes.com/editorial/10-best-revolvers-time/99397",
        "https://www.pewpewtactical.com/best-beginner-revolvers/",
        "https://en.wikipedia.org/wiki/Colt_Python",
        "https://www.handgunsmag.com/editorial/colt_python_complete_history/138916",
        "https://www.shootingtimes.com/editorial/original-colt-python-review/510644",
        "https://www.americanrifleman.org/content/colt-s-pythons-then-now/",
        "https://coltfever.com/python/",
        "https://en.wikipedia.org/wiki/Category:Smith_%26_Wesson_revolvers",
        "https://www.sportsmans.com/ammo-caliber-size-chart?srsltid=AfmBOoraf0bNfnansm-UrJwJBUbzOyZhyE9TAcpPkbLCv_8ACvhr7QAE",
        "https://en.wikipedia.org/wiki/Caliber",
        "https://en.wikipedia.org/wiki/Kalashnikov_rifle",
        "https://www.pewpewtactical.com/best-ak-47/",
        "https://gundigest.com/rifles/the-best-ak-47-rifles-you-can-find-in-the-u-s",
        "https://en.wikipedia.org/wiki/List_of_firearm_brands",
        "https://en.wikipedia.org/wiki/Sturm,_Ruger_%26_Co.",
        "https://en.wikipedia.org/wiki/Smith_%26_Wesson",
        "https://en.wikipedia.org/wiki/Colt%27s_Manufacturing_Company",
        "https://en.wikipedia.org/wiki/Remington_Arms",
        "https://en.wikipedia.org/wiki/Springfield_Armory,_Inc.",
        "https://en.wikipedia.org/wiki/FN_Herstal",
        "https://en.wikipedia.org/wiki/Heckler_%26_Koch",
        "https://en.wikipedia.org/wiki/Daniel_Defense",
        "https://en.wikipedia.org/wiki/Barrett_Firearms",
        "https://en.wikipedia.org/wiki/Česká_zbrojovka_Uherský_Brod",
        "https://en.wikipedia.org/wiki/Taurus_Arms",
        "https://en.wikipedia.org/wiki/Sig_Sauer",
        "https://en.wikipedia.org/wiki/Walther_Arms",
        "https://en.wikipedia.org/wiki/Savage_Arms",
        "https://en.wikipedia.org/wiki/Browning_Arms_Company",
        "https://en.wikipedia.org/wiki/Beretta",
        "https://en.wikipedia.org/wiki/Benelli_Armi",
        "https://en.wikipedia.org/wiki/Franchi_(firearms)",
        "https://en.wikipedia.org/wiki/Steyr_Arms",
        "https://en.wikipedia.org/wiki/Rößler_Waffen",
        "https://en.wikipedia.org/wiki/Zastava_Arms",
        "https://en.wikipedia.org/wiki/Kalashnikov_Concern",
        "https://en.wikipedia.org/wiki/IMBEL",
        "https://en.wikipedia.org/wiki/Izhmekh",
        "https://en.wikipedia.org/wiki/Tula_Arms_Plant",
        "https://en.wikipedia.org/wiki/Norinco",
        "https://en.wikipedia.org/wiki/Bersa",
        "https://en.wikipedia.org/wiki/Israel_Weapon_Industries",
        "https://en.wikipedia.org/wiki/Sako",
        "https://en.wikipedia.org/wiki/Steyr_Mannlicher",
        "https://en.wikipedia.org/wiki/Palmetto_State_Armory",
        "https://en.wikipedia.org/wiki/Winchester_Repeating_Arms",
        "https://en.wikipedia.org/wiki/Bravo_Company_Manufacturing",
        "https://en.wikipedia.org/wiki/Strayer_Voigt_Inc",
        "https://staccato2011.com/",
        "https://en.wikipedia.org/wiki/Les_Baer_Firearms",
        "https://en.wikipedia.org/wiki/Wilson_Combat",
        "https://en.wikipedia.org/wiki/Nighthawk_Custom_Pistols",
        "https://en.wikipedia.org/wiki/Ed_Brown_Products",
        "https://en.wikipedia.org/wiki/BUL_Armory",
        "https://atlasgunworks.com/"
    ]

    downloader = WebsiteDownloader(output_dir="downloaded_sites")
    downloader.crawl(websites, max_depth=1)
