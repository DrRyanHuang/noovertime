import requests as r
from lxml import html
from typing import Callable
from collections import namedtuple
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import time


HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.gushici.net"

Writer = namedtuple("Writer", ["name", "url", "info", "poems"])

SAVE_DIR = "./poem"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_once(func: Callable):
    """
    使函数只运行一次，并将结果缓存下来，之后再次调用时直接返回缓存结果。

    Args:
        func (Callable): 需要被装饰的函数，无返回值或返回任意类型均可。

    Returns:
        Callable: 返回一个新的函数，该函数只会在第一次调用时执行原始函数，并将结果缓存下来。
            之后再次调用时直接返回缓存结果。
    # https://github.com/Obmutescence/4DPocket/blob/main/python/utils/decorator.py
    """

    def wrapper(*args, **kwargs):
        if not wrapper._has_run:
            wrapper._has_run = True
            wrapper._result = func(*args, **kwargs)
        return wrapper._result

    wrapper._has_run = False
    return wrapper


@run_once
def get_writer_list():
    # 爬取这个网站有的诗人列表

    BASE_WRITER_URL = f"{BASE_URL}/zuozhe/"
    response = r.get(BASE_WRITER_URL, headers=HEADERS)
    response.encoding = response.apparent_encoding
    if response.status_code != 200:
        raise Exception(
            "Request failed with status code {}".format(response.status_code)
        )

    tree = html.fromstring(response.text)
    write_xpath = "/html/body/div[2]/div[2]/div/div[2]"
    box_div = tree.xpath(write_xpath)[0]  # 使用你提供的XPath

    writer_list = []
    for href in box_div:
        writer_name = href.text
        link_basename = href.attrib.get("href", None)
        writer_url = f"{BASE_URL}{link_basename}"
        writer = Writer(name=writer_name, url=writer_url, info=None, poems=None)
        writer_list.append(writer)

    return writer_list[:-1]  # remove `更多>>`


def get_writer_info(writer_url: str):
    # 爬取诗人信息

    response = r.get(writer_url, headers=HEADERS)
    response.encoding = response.apparent_encoding
    if response.status_code != 200:
        raise Exception(
            "Request failed with status code {}".format(response.status_code)
        )

    soup = BeautifulSoup(response.text, "lxml")

    info = soup.find("div", class_="left")

    poems_url_basename = soup.find("div", class_="arti").a.attrs["href"]
    poems_url = f"{BASE_URL}/{poems_url_basename}"

    return info, poems_url


def get_writer_poem_list(poems_url: str):
    # 爬取诗人作品列表

    poems_list = []
    while True:

        response = r.get(poems_url, headers=HEADERS)
        response.encoding = response.apparent_encoding
        if response.status_code != 200:
            raise Exception(
                "Request failed with status code {}".format(
                    response.status_code
                )
            )

        soup = BeautifulSoup(response.text, "lxml")

        divs = soup.find_all("div", class_="gushici")
        poems_list += divs

        next_page_a = soup.find("div", class_="pagesleft").find(
            "a", class_="amore"
        )

        if next_page_a is None:
            break
        next_page_basename_url = next_page_a.attrs["href"]

        poems_url = f"{BASE_URL}/{next_page_basename_url}"

    return poems_list


def _get_url_poem_div(poem_div):
    # 从 div 解析出 url

    poem_basename_url = poem_div.div.p.a.attrs.get("href")
    poem_url = f"{BASE_URL}/{poem_basename_url}"
    return poem_url


def get_poem_info(poem_url: str):
    # 爬取诗歌信息

    response = r.get(poem_url, headers=HEADERS)
    response.encoding = response.apparent_encoding
    if response.status_code != 200:
        raise Exception(
            "Request failed with status code {}".format(response.status_code)
        )

    soup = BeautifulSoup(response.text, "lxml")

    soup_left = soup.find("div", class_="left")

    save_info = [soup_left.find("div", class_="gushici")]
    save_info += soup_left.find_all("div", class_="shici")

    return save_info


def _get_poem_output(div_list: list):
    return "".join([div.text for div in div_list])


def _handle_special_title(poem_title: str):
    poem_title = poem_title.replace(" / ", "_").replace("/", "_")
    return poem_title


if __name__ == "__main__":

    for writer in tqdm(get_writer_list()):

        writer_dir = os.path.join(SAVE_DIR, writer.name)
        # if os.path.exists(writer_dir):
        #     continue
        os.makedirs(writer_dir, exist_ok=True)

        info, poems_url = get_writer_info(
            writer.url,
        )

        # writer.info = info
        poems_list = get_writer_poem_list(poems_url)

        for poem_div in poems_list:

            # 诗歌标题
            poem_title = poem_div.p.text
            poem_title = _handle_special_title(poem_title)
            # 诗歌链接
            poem_url = _get_url_poem_div(poem_div)

            poem_info_div_list = get_poem_info(poem_url)
            poem_str = _get_poem_output(poem_info_div_list)

            # 写入文件
            poem_file = os.path.join(writer_dir, poem_title) + ".txt"
            with open(poem_file, "w", encoding="utf8") as f:
                f.write(poem_str)

            # time.sleep(0.5)
