import pandas as pd
from scipy import spatial 
import ast

def gendf(filename):
    with open(filename) as f:
        contents = f.read().split("文件名：")[1:]
        df = pd.DataFrame(contents, columns=["content"])
        # save
        df.to_csv(filename.replace(".txt", ".csv"), index=False)
    return df

EMBEDDING_MODEL = "text-embedding-3-small"

# from openai import OpenAI
# client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   emb = client.embeddings.create(input = [text], model=model).data[0].embedding
   return emb

def process_qa():
    qadf = pd.read_csv("dataset/qa.csv")
    # rename columns
    qadf.rename(columns={"问题": "question", "回答": "answer"}, inplace=True)
    qadf['embedding'] = qadf.question.apply(lambda x: get_embedding(x))
    qadf.to_csv("dataset/qa-with-embedding.csv", index=False)

qadf = pd.read_csv("dataset/qa-with-embedding.csv")
qadf['embedding'] = qadf['embedding'].apply(ast.literal_eval)

df = pd.read_csv("dataset/leader-talk-with-embedding.csv")
df['embedding'] = df['embedding'].apply(ast.literal_eval)


def strings_ranked_by_relatedness(
    # query: str,
    query_embedding,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    strings_and_relatednesses = [
        (row["content"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]



for i, row in qadf.iterrows():
    input_str = row["question"]
    # emb = get_embedding(input_str)
    emb = row["embedding"]
    strings, relatednesses = strings_ranked_by_relatedness(emb, df, top_n=1)
    content_str = ""
    for s, r in zip(strings, relatednesses):
        content_str += s + "\n"
    content_str += input_str + "\n"
    with open(f"dataset/ps/{i}.txt", "w") as f:
        f.write(content_str)




# df['embedding'] = df.content.apply(lambda x: get_embedding(x))

# df.to_csv("dataset/leader-talk-with-embedding.csv", index=False)