import polars as pl
import time
import os


base_data_dir = '/data/profits_bot'
dumps_dir = 'decomp_dumps'
out_dir = 'processed_70'
quanitle = 0.3 # this is the minimum quanitile that will get selected.

def process(filename):
    try:
        file_path = f'{base_data_dir}/{dumps_dir}/{filename}'

        wsb_df = pl.scan_ndjson(
            file_path,
            batch_size=100000,
        ).select(
            pl.col(['id', 'url', 'num_comments', 'subreddit', 'title', 'selftext', 'score'])
        ).filter(
            (pl.col("score") > 1) &
            (~pl.col('title').str.contains('Daily Discussion Thread|Earnings for the Week|Weekend Discussion Thread|What Are Your Moves Tomorrow')) &  # ToDo: this is for WSB only
            (pl.col('selftext') != '[deleted]') &
            (pl.col('selftext') != '[removed]') &
            (pl.col('selftext').str.n_chars() > 150)
        )
        
        # some basic stats
        stats_df = wsb_df.select([
            pl.max('score').alias('max_score'),
            pl.min('score').alias('min_score'),
            pl.count('score').alias('total scores'),
            pl.quantile('score', 0.3, 'higher').alias('70th_score_quantile'),
            pl.var('score').alias('score_variance'),
            pl.col('score').skew().alias('score_scew'),
            pl.std('score').alias('score_std'),
            pl.mean('score').alias('score_mean'),
        ]).collect()
        
        # these stats are used later for normalizating posts
        max_score = stats_df[0, 0]
        min_score = stats_df[0, 1]
        total_rows = stats_df[0, 2]
        low_score_to_get = stats_df[0, 3]
        score_std = stats_df[0, 6]
        score_mean = stats_df[0, 7]

        quantile_70th = stats_df[0, 3]

        subreddit = wsb_df.collect().head(1)['subreddit'][0]
        
        # select everything with a score above the 70th quantile

        wsb_df = wsb_df.filter(
            pl.col('score') > int(quantile_70th)
        ).with_columns(
            ((pl.col('score') - float(score_mean)) / float(score_std)).alias('z_score'),
            ((pl.col('score') - float(min_score)) / (float(max_score) - float(min_score))).alias('normalized_score')
        ).select(
            pl.col(['id', 'title', 'selftext', 'z_score', 'normalized_score','subreddit'])
        )

        out_filename = f'{base_data_dir}/{out_dir}/{subreddit}_posts.jsonl'
        with open(out_filename, mode="ab") as f:
            wsb_df.collect().write_ndjson(f)
        
        length = wsb_df.collect().shape[0]
        return (length, out_filename)
    except Exception as e:
        print(f'could process file {filename}. got error: {e}')
        return (0, '')

if __name__ == '__main__':
    
    for file in os.listdir(f'{base_data_dir}/{dumps_dir}'):
        filename = os.fsdecode(file)
        if 'submissions' in filename:
            start = time.time()
            print(filename)
            print(f'parsing {filename}')
            res = process(filename)
            print(f'wrote {res[0]} lines in {time.time() - start}')