"""
Download the compressed datadumps from https://the-eye.eu/redarcs/. Organized by subreddit.
"""
import parfive


downloader = parfive.Downloader()
download_path = "/data/profits_bot/raw_dumps/"
file_list = './raws_to_download.txt'

with open(file_list) as file:
    urls = [line.rstrip() for line in file]

urls = list(set(urls))  # removes any duplicates

for url in urls:
    downloader.enqueue_file(url, path=download_path, overwrite=False)
    
    
files = downloader.download()


# ToDo: catch errors and retry, make this recurse X times until all files are downloaded
if files.errors:
    breakpoint()
    print(files)
    # files = downloader.retry(files)
    pass

print(files)