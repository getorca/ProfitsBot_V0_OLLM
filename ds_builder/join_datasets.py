import polars as pl
import os
import time
import gc



base_data_dir = '/data/profits_bot'
out_dir = 'processed_70'


def process_sub(sub_name):
    '''
    Joins the comments and posts and outputs [subname_joined.jsonl]
    - also normailzes post scores
    '''
    
    comments_file = f'{base_data_dir}/{out_dir}/{sub_name}_comments.jsonl'
    posts_file = f'{base_data_dir}/{out_dir}/{sub_name}_posts.jsonl'
    out_file = f'{base_data_dir}/{out_dir}/{sub_name}_joined.jsonl'
    
    c_df = pl.scan_ndjson(
        comments_file,
    )


    p_df = pl.scan_ndjson(
        posts_file
    )
    
    jdf = p_df.join(c_df, left_on='id', right_on='link_id', how='inner', suffix='_comment')
        
    jdf = jdf.explode('body', 'score', 'id_comment')

    stats_df = jdf.select([
        pl.max('score').alias('max_score'),
        pl.min('score').alias('min_score'),
    ]).collect()

    # these stats are used later for normalizating posts
    max_score = stats_df[0, 0]
    min_score = stats_df[0, 1]


    jdf = jdf.with_columns(
        ((pl.col('score') - float(min_score)) / (float(max_score) - float(min_score))).alias('comment_normalized_score')
    ).select(
        pl.col(['id', 'title', 'selftext', 'z_score', 'normalized_score','subreddit', 'body', 'comment_normalized_score'])
    ).collect()
    
    with open(out_file, mode="ab") as f:
        jdf.write_ndjson(f)
    
    length = jdf.shape[0]
    return (length, out_file)    

def merge_ds():
    
    out_file = f'{base_data_dir}/{out_dir}/top.jsonl'
    
    mdf = pl.scan_ndjson(
        f'{base_data_dir}/{out_dir}/*_joined.jsonl'
    ).with_columns(
        (pl.col('normalized_score') + pl.col('comment_normalized_score')).alias('combined_score')
    ).sort(
        ['combined_score'], descending=True
    ).limit(250000).collect()
    
    with open(out_file, mode="ab") as f:
        mdf.write_ndjson(f)
        
    return True
    

if __name__ == '__main__':
    
    # Step 3.1 - join the datasets and comments
    for file in os.listdir(f'{base_data_dir}/{out_dir}'):
        filename = os.fsdecode(file)
        if '_posts' in filename:
            print(f'processing {filename}')
            sub_name = file.rsplit('_', 1)[0]
            if not os.path.isfile(f'{base_data_dir}/{out_dir}/{sub_name}_joined.jsonl'):
                s1_res = process_sub(sub_name) 
                print(f'wrote {s1_res[0]} to file {s1_res[1]}')
            
    # Step 3.2 - join the joins
    merge_ds()
    print('merged_dataset')
    
    # for file in os.listdir(f'{base_data_dir}/{dumps_dir}'):
    #     filename = os.fsdecode(file)
    #     files = []
    #     if '_joined' in filename:
    #         files.append(filename)
