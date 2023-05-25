import polars as pl
import os
import time
import gc


comments_chunk_directory = './data/chunk'
comments_out_file = f'./data/comments_out_{str(time.time())}.jsonl'


base_data_dir = '/data/profits_bot'
dumps_dir = 'decomp_dumps'
out_dir = 'processed_70'


def parse_chunk(filename):
    
    file = f'{base_data_dir}/{dumps_dir}/{filename}'
    
    wsb_df = pl.scan_ndjson(
        file,
        batch_size=500000,
        # n_rows=2000000,
        row_count_name= 'row_count',
        ).select(
            pl.col(['parent_id', 'link_id', 'body', 'score', 'id', 'subreddit'])
        ).filter((
            (pl.col('parent_id') == pl.col('link_id')) &  # maybe need to confirm....
            (pl.col('body') != '[deleted]') &
            (pl.col('body') != '[removed]') &
            (pl.col('score') > 3) &
            (pl.col('body').str.n_chars() > 200)
    ))

    wsb_df = wsb_df.sort(['link_id', 'score'], descending=True)

    wsb_df = wsb_df.groupby("link_id", maintain_order=True).agg([
        pl.col('subreddit').first(),
        pl.col('body').head(3),
        pl.col('score').head(3),
        pl.col('id').head(3),
        pl.col('id').count().alias('comment_count'),    
    ])

    wsb_df = wsb_df.with_columns(pl.col('link_id').str.slice(3).alias('link_id')) 
    
    df = wsb_df.collect()
    
    subreddit = df[0]['subreddit'][0]
    
    out_filename = f'{base_data_dir}/{out_dir}/{subreddit}_comments.jsonl'
    with open(out_filename, mode="ab") as f:
        df.write_ndjson(f)
    
    count = df.select(pl.count())[0, 0]  
      
    del df
    del wsb_df
    gc.collect()
    
    return count

    
def loop_comment_chunks():
    '''
    If the comments are chunked this function can loop over them. Chunking is helpfull if the dataset is to large for memory.
    To Chunk comments use: `jq -c . < wallstreetbets_comments.jsonl | split -l 5000000 --additional-suffix=.jsonl - ./chunk/comments_`
    '''
    
    for file in os.listdir(comments_chunk_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jsonl"):
            start = time.time() 
            file = os.path.join(comments_chunk_directory, filename)
            print(f'parsing {file}')
            res = parse_chunk(file)
            print(f'wrote {res} lines in {time.time() - start}')
        else:
            continue

        
if __name__ == '__main__':
    
    for file in os.listdir(f'{base_data_dir}/{dumps_dir}'):
        filename = os.fsdecode(file)
        if 'comments' in filename:
            start = time.time() 
            # file = './data/wallstreetbets_comments.jsonl'
            print(f'parsing {filename}')
            res = parse_chunk(filename)
            print(f'wrote {res} lines in {time.time() - start}')
    
    # loop_comment_chunks()