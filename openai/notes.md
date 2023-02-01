# Playground

No code required, just need to login. Initially free, but after 3 months or a number of tokens (whatever comes first) user must pay to send other requests.

choose mode: completion, insert, and edit

model: davinci, ada, and others. Pricing is different depending on model

View code: there is an option at the top that allows you to check the code in language of choice, such as python, node.js, curl, or check json file

"Best of" function: generates multiple answers (be careful when getting billed)

# Prompt Engineering

## Simple questions

* Provide sentence with a question mark, or fullstop, so the model knows how to respond (not to complete)
Ex: What color are fire engines in the UK?

## Generate some ideas

* Use colon after a statement
Ex: Brainstorm some ideas combining VR and fitness:

## Simple completion
* Leave sentence unfinished and model will continue from where we stopped. Be carefull by number of tokens generated
Ex: Vertical farming provides a novel solution for

## Classification
Provide a sentence and ask GPT to classify it.
Ex: Tweet: I loved the new Batman movie!
Sentiment:

## Multiple Requests
Provide a list of items.
Ex: Classify the sentiment in these tweets:
1. "I can't stand homework"
2. "This sucks, I'm bored"
3. "I can't wait for Halloween!!!"

Tweet sentiment ratings:
1. 

## Grammar correction
Ex: Correct this to standard english:
She not want to sleep.

## Summarization
Ex: Summarize this text for a child:

bla bla bla 

## Translation
Language translation, Codex (Code translation)

# Improving GPT prompts to get better responses
Be explicit
 - Show and tell:
 Write a poem about OpeanAI vs. Write a short inspiring poeam about OpenAI, focusing on the recent DALL-E product launch (DALL-E is a text to image ML model) in the style of Shakespeare
 
 - Give more example responses:
 Extract the important entities mentioned in the text below. First extract all copmany names, then extract all people names, then extract specific topics which fit the content and finally extract general overarching themes.
 
 Desired format:
 Company names: -,-
 People names: -,-
 Specific topics: -,-
 General themes: -,-
 
 Text: {text}
 
 - Reduce fluff (be precise)
 
 - Positive instruction:
 Tell the AI what to do instead of what NOT to do
 
 - Lead the way:
 For example, in Codex, provide some lines with specific programming language.
 
 - Adjust tone by adding tags such as polite response
  
 - Provide quality data
  
 - Use "temperature" and "top_p" settings":
 Temperature is high, the AI can choose freely the words (helpful for creative writing), avoid conversation drift
 Top_p to avoid long tail
 
 - Position of \n matters (completion or generated a new answer, such as in a dialogue)
 
 - Create smaller tasks (create outline, write intro paragraphs, write a paragraph about..., etc) vs. one large task (write an article about...)

# Prompt Design - Templating
Enumerate tasks to help the model. Example:
"""
1. You are an assistant helping...
2. ...to generate a <insert task here, e.g. write a blog post>...
3. ...based on a prompt (e.g. use this format, replacing text in brackets with the result. Do not include the brackets in the output.)

4. Blog post:
5. #[Title of blog post]
[Content based on the prompt]
6. Prompt itself followed by a new line.
"""

Example:
"""
You are an assistant helping to generate a blog post based on a prompt. Use this format, replacing text in brackets with the result. Do not include the brackets in the output:

Blog post:
# [Title of blog post]
[Content of blog post]
"""
Blue Snowball\n

## Summarizing text
Summarize this for a 3rd grader. 
"""
Provide instructions inside the comment
"""

# Code examples 
## Simple CURL example
```
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API" \
  -d '{
  "model": "text-davinci-003",
  "prompt": "Say this is a test",
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 1,
  "frequency_penalty": 0,
  "presence_penalty": 0
}'
```

## Simple Python Example
- Install the library (`pip install openai`)
```py
import os
import openai

openai.api_key = os.getenv('OPENAI_API')

response = openai.Completion.create(model='text-davinci-003', prompt='Say this is a test', temperature=0, max_tokens=7)
```

## Simple Node.js example
- Install the library (`npm install openai`)
```
const { Configuration, OpenAIApi } = require("openai");

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const response = await openai.createCompletion({
  model: "text-davinci-003",
  prompt: "Say this is a test",
  temperature: 0,
  max_tokens: 7,
});
```

## Simple C# example
- Install the library (`Install-Package OpenAI`)
```
var api = new OpenAI_API.OpenAIAPI("sk-myapikeyhere", Engine.Davinci);

var result = await api.Completions.CreateCompletionAsync("Say this is a test", temperature: 0);

// should print something like "\n\n This is indeed a test"
Cnosole.WriteLine(result.ToString());
```

## Powershell example
```
$apiEndpoint = "https://api.openai.com/v1/completions"
#apiKey = "sk-APIKEY" 

$headers = @{
  Authorization = "Bearer $apiKey"
}

$contenttype = "application/json"

$body = @{
  model = "text-davinci-003"
  prompt = "This is only a test"
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri $apiEndpoint -Method POST -Headers $headers -Body $body -ContentType $contenttype

$generatedText = $response.Content | ConvertFrom-Json

Write-Output $generatedText.choices[0].text
```

# The Moderation Endpoint
This is the only free-to-use endpoint and can be used for completions.
`POST https://api.openai.com/v1/moderations/`

Need to provide some text ('input'). The moderations endpoint looks at how harmful the input could be to the user. There are different categories, such as hate, self-harm, sexual, etc. 

Check out the `moderation.ipynb` notebook for an example. 

# Completion parameters
## How tokens work
Each token is given a probability score by the AI, and response is created by joining tokens together that have high probabilities (this can be adjusted with code).

## Calling the API

```
response = openai.Completion.create(
 model="text-davinci-003",
 prompt="What is human life expectancy in the US?",
 max_tokens=500,
 temperature=0.1
)
```
We can adjust the request by changing parameters such as `max_tokens` or `temperature`.
* `max_tokens`: Only cuts the output once the max value is reached, also counts input/query tokens
* `top_p`: default is 1, 0.9 allows AI to be more creative, 0.3 consider only the top 30% by mass.
* `temperature`: 0 for well defined answers (token with highest probability), 0.9 for creative applications, 2 would give all tokens same probability 
* `n`: generate multiple answers (completions), default is 1 - be careful as multiple responses will consume many tokens
* `stop`: preserves tokens by avoiding verbose responses, defaults to null. Examples: stop=['\n', 'Human:', 'AI:']
* `best_of`: generates multiple answers on server end, returns the best one (default is 1). This can be combine with `n` to send more than 1 response. This also counts towards the tokens. 
* `suffix`: Tell the AI what will follow, limited to 40 characters. 
* `echo`: include the prompt in the completion (default is false, so doesn't include text that was used in the prompt).
* `user`: optional, useful for policy violations.

# Adjusting probabilities 
Each token represents a word, or part of a word. There are about to 50,000 tokens in GPT-3. We can check the tokens ![here](https://beta.openai.com/tokenizer).

## logprobs
We can ask the model what are the highest probability tokens within the response. The tokens are given with a log of the probability (logprob), so the close to 0, the higher the probability.

## Banning tokens - logit_bias
Assign a Bias by deducting 100 (-100) from a token's logprob. Example:
```
response = openai.Completion.create(
 model='text-davinci-003',
 prompt='What is the capital of France?',
 logit_bias: {'6342': -100}
)
```
This would change the logprob for token_id = 6342. We could also address this by adding (instead of deducting) a value to the logprob of the token of interest. 

## Dealing with repetition
* `presence_penalty`: a way to adjust the chances of a token being reused in a response. It doesn't matter how many times a word has been used, it will be penalized by the same amount throughout the response.
* `frequency_penalty`: cumulative, so the more a token is used, the less likely it is to appear again. 
Default for both is 0, range is -2 to 2.

# Coding for Edits
`POST https://api.openai.com/v1/edits`
- Uses a different model than `Completion`, with input and instruction.

```
openai.Edit.create(
 model='text-davinci-edit-001',
 input='What day of the eek is it?',
 instruction='Fix the spelling mistakes'
)
```
Response is a json file. We can add other parameters such as top_p, n, to edit the responses. 

# Coding to create images - DALL-E
Creates images from text. Images are always squared and in PNG format. 

## Create a new image example
- `prompt`: used to describe the image we require. 
- `size`: 256x256, 512x512, 1024x1024
- `response_format`: Can be returned as a URL 
- `n`: generate 20 images per minute
- `user` :for identification/security purposes

`POST https://api.openai.com/v1/images/generations`

```
response = openai.Image.create(
 prompt='An armchair in the shape of an Avocado',
 n=1,
 size='1024x1024'
)
```
Response will give a temporary URL valid for an hour - save it before it expires at this costs tokens.

## Modifying images
`POST https://api.openai.com/v1/images/edits`
Edit existing images. The need to be in an appropriate size (<4MB) and in PNG format. Also requires a mask image, which would be the same image with an area of it erased. 

```
response = openai.Image.create_edit(
 image=open('sunlit_lounge.png', 'rb'),
 mask=open('mask.png', 'rb'),
 prompt='A sunlit indoor lounge area with a pool containing a flamingo',
 n=1,
 size='1024x1024'
)

image_url = response['data'][0]['url']
```

## Variations of an image
`https://api.openai.com/v1/images/variations`
Variation of an image. Provide a source image (PNG), size and response_format. 

## Tips for handling images in Python
Convert an image to the right dimensions. Load an image first, convert it to byte stream and then into a byte array before feeding to the model.

```
from io import BytesIO
from PIL import image

# Read the image file from disk and resize it
image = Image.open('image.png')
width, height = 256, 256
image = image.resize((width, height))

# This is the BytesIO object that contains the image
byte_stream = BytesIO()
image.save(byte_stream, format='PNG')
byte_array = byte_stream.getvalue()

response = openai.Image.create_variation(
 image=byte_array,
 n=1,
 size='1024x1024'
)
```

## Image moderation
If an image violates the OpenAI policy, an error will be sent back.

# Codex
GPT has been trained on different GitHub repositories. It can turn comments into code, complete lines or functions, refer to different libraries or API calls, add comments, and rewrite code. 

Same endpoints (`completion` and `edits`), but different models: `code-davinci-002`, `code-cushman-001`.

## Writing codex
Use comment language such as:
```
"""
Ask the user their name and say hello.
"""
```

## Generating SQL
```
""" 
Comment with table name, columns, and instructions, such as 'Create a MySQL query for all customers in Texas named Jane
"""
query = <AI will fill this>

run_query(query) # use this to stop the AI from using too many tokens
```

## Exlpaining code
Include function commented at the top, then ask AI for explanation in human readable form. 

## Best practice
- Start prompt with a comment (`//`, `#` or `"""`)
- Finish code wit ha question (as a comment)
- Specify the language in the first comment (i.e. # Python language)
- Tell codex wich libraries to use
- Lower temperature close to 0 to get more precise results
- Be as precise as possible in comments, specify libraries and variables that we want to use

## Fixing errors
```
<Code for a broken function>

"""
Explain why the previous function doesn't work.
"""
```

## Converting between languages
```
# Convert this from Python to R
# Python Version

<Python code>

# End

# R Version
```

## Building code
Using the `edits` endpoint, we can provide a function or blank as an input, and then instructions on what to do, such as 'Include documentation'.

# Fine-tuning
`https://api.openai.com/v1/fine-tunes`

Once the model has been fine tuned, we don't need to provide as many examples anymore, thus saving on costs and making the process more efficient. 

- Prepare a training data file (JSONL)
- Upload the file
- Tell the model to import the file
- Use the new model for fine-tuned responses
- Consider training a lower cost model to save on money **(ada instead of davinci if we don't need the conversation to go in every direction, but rather focused on the topic/application)**.

## Preparing Training Data
```
{"prompt": "<prompt text>", "completion": "<ideal generated text"}
{"prompt": "<prompt text>", "completion": "<ideal generated text"}
{"prompt": "<prompt text>", "completion": "<ideal generated text"}
...
```

- More than a couple hundred sets work best
- Increasing examples is the best way to improve performance

## Different types of training
There are different ways we train the model during fine tuning - classification, summarize, expand, extract. In all cases, we need to provide a training data set (JSONL file). In this file, include a file with a prompt, followed by a clear marker showing the end of the prompt, and then providing a completion (preceded by a space). 
- Need to keep in mind the token limit for different models!

## Training to expand
Design idea: create sales description from product properties
- prompt: `"<properties>\n\n###\n\n"`
- completion: `"<sales descriptions>END"`

## Extract entities from text
- Extract 3 entities from an email ,such as a name, email address, phone number
- prompt: "<any text, for example news article>\n\n###\n\n"
- completion : " <extracted data> END"
- It's **key** to stay consistent with how we format the prompt and the completion.

## Chatbots
- No memory of previous answers
- Create a fine tuning file with examples of how the bot behaves
- Can easily go off track, need to keep that in mind for cost and also to add restrictions

### Design idea - customer service chatbot
- prompt should include a summary of the interaction so far, specific information, and then an example of the dialogue
```
"{Summary: <summary of the interaction so far>\n\nSpecific information: <e.g. order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\n\nCustomer: <message2>\nAgent:"
```
Each customer example needs a custom response. 

- completion: " <response2>\n\n"

Best results are usually obtained when using real-world data. Only the last response should be in the completion part.

### Large documents or text
Because of the maximum number of tokens, it is hard to handle large documents. 
- Embedding is a better option

### Final tips
- The separator (in this course `\n\n###\n\n`) must not appear in the body of the prompt
- Always start the completion with a space - this improves model performance
- Make sure prompt ends with a label and a column
- Use Stop text (END or\n) to prevent AI from adding random test
- Upper and lower case matter a lot! Use it when it is correct, such as the beginning of a sentence or in city name.

# Generating fine-tuning datasets
Instead of creating these manually, we could scrape data off the internet that could be used for fine-tuning. 

Another alternative would be to use GPT to generate training data. For example, use the more advanced Davinci model to train the ADA model on a focused topic. We could also write a code to first write questions for a text, and then use GPT to answer these questions. You can then store the data as a json file to be used during fine tuning. 

# Uploading and processing fine tuning files
Use the method:

```
openai.File.create(
 file=open("mydata.jsonl", "rb"),
 purpose="fine-tune"
)
```
ID: uploaded file gets an ID, we need to keep track of it so we can use it when it comes to trigerring the training

Once file has been uploaded, we can use it to fine tune the model:
```
openai.FineTune.create(
 training_file='<file ID generated at upload>',
 model='ada',
 suffix='chatbot'
)
```
We only need to provide the base name of the model, not '001'. We can also use the name of our existing model to the model parameter. Suffix can be used to make it easier to identify the model. 

## Unicode errors
The model may not recognize emojis and such. In that case, we will get a `UnicodeEncodeError`. 

# Embedding, searching and integrating large documents and text
Embedding: representing words as a string of floats, which is then converted to a vector that can be represented in a multidimensional space. Similar words will have similar coordnates (thus will be closer in space), while different words will have more different coordinates (and will be farther apart in space). 

## Embedding engine
We can use the `text-embedding-ada-002`. This allows us to use up to 8192 characters. 

## Creating first embedding vector
Example: "she went fishing"

```
response = openai.Embedding.create(
 input="she went fishing",
 model="text-embedding-ada-002"
)
```

This return a large list of floating points 
- 1536 numbers for ada-002

- `response['data'][0]['embedding']`

- embedding: compare two or more strings and see how similar they are

## Libraries used during embedding
- Work with CSV files for the course
- Pandas, NumPy, OpenAI
- Also use `from openai.embeddings_utils import get_embedding, cosine_similarity`
- Cosine similarity is handy when comparing how similar two strings are

## Text similarity
- Get the vectors for multiple blocks of text:

```
import openai

response = openai.Embedding.create(
 input=[
 	"Eating food",
 	"I am hungry", 
 	"I am traveling", 
 	"Exploring new places"],
 model="text-embedding-ada-002"
)
```
We can read the vectors from the response object. For example, the respective vectors for the strings above are:
```
vector1 = response['data'][0]['embedding']
vector2 = response['data'][1]['embedding']
vector3 = response['data'][2]['embedding']
vector4 = response['data'][3]['embedding']
```

### Calculating the distance
To find the distance between two phrases, we need to calculate the "dot product" of the two vectors

```
import numpy as np

distance = np.dot(vector1, vector2)
```

This will give a float value, such as 0.87365. WE can then compare it with the dot product between different phrases (make sure to reuse one of the phrases so we can compare the results). The new distance value is 0.78477. The closer the value is to 1, the more similar the two sentences are.

## Embedding data
Good to use local system when doing this. But how do we perform this on a document or blocks of text?
- Semantic search: use smaller blcks of text, vectorize it, and then compare the blocks for similarity. 

### Preprocessing the data
For embedding these blocks of text, we first need to preprocess the data. In that case, it is useful to use: 
- `pandas`: load the dataset as a dataframe
	- Actions: rename columns, handling of missing values, combining features (such as `Title` and `Summary` columns into a single new feature `Combined`)
- library to count tokens (e.g. `transformers`): useful to make sure we have the right number of tokens and don't break the model
	```
	from transformers import GPT2TokenizerFast
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	```
	- Create a column with number of tokens (n_tokens):
	```
	df[n_tokens'] = df['Combined'].apply(lambda x: len(tokenizer.encode(x)))
	```
	- Only keep reviews that aren't over the max token limit:
	```
	df = df[df['n_tokens'] < 8000]
	```

### Get embeddings and save
To get embeddings, we apply a lambda function to our pandas DataFrame calling the get_embedding method
```
df['ada_vector'] = df['Combined'].apply(
 lambda x: get_embedding(x, engine='text-embedding-ada-002')
)

df.to_csv('data/embedded_reviews.csv', index=False)
```

```
def get_embedding(text, model='text-embedding-ada-002'):
	# Import library
	import openai

	# Swap \n for spaces
	text = text.replace('\n', ' ')
	
	# Call the API with the parameters
	# return ['data'][0]['embedding']
	# This is the 2D vector for the text
	return openai.Embedding.create(
	 input=[text],
	 model=model)['data'][0]['embedding']
```

### Using saved data
```
import pandas as pd
import numpy as np

# read the csv output file
df = pd.read_csv('output/embedded_reviews.csv')

# Convert the strings stored in the ada_vector column into vector objects
df['ada_vector'] = df['ada_vector'].apply(eval).apply(np.array)

# We can access the values using 
df.loc[0]['ada_vector']
```

### Define a semantic search function
Get an embedding vector for the users question or text query

```
# convert our search term into a vector
searchvector = get_embedding(
 'delicious beans', 
 model='text-embedding-ada-002'
)

# create a new column using cosine similarity on every row
# comparing our searchvector string to the value in our local dataset
df['similarities'] = df['ada_vector'].apply(
 				lamda x: cosine_similarity(x, searchvector))

```

**Cosine similarity**: also called cosine distance. The closer two words or items are to each other, the smaller the angle between them. That means, the smaller the angle, the more similar the items. Therefore, if the cos similarity is close to 1, the words are very similar, and if close to 0, the words are assimilar.  

```
# Sort the values by the cosine similarity and take the top n rows
# response is up to 3 rows, it contains all the data
res = df.sort_values('similarities', ascending=False).head(3)

# we can access the combined column to get the matching text
# we could also access any of the other columns in the DataFrame
res.loc[0]['combined]
```

### Combining other libraries with GPT
Other libraries with word embedding capability can be used for embedding:
- `Word2Vec`
- `Glove`
- `BERT`
- `Fasttext`

All have their own pros and cons.

Procedure when working with GPT and Word2Vec
- Use Word2Vec to generate vectors
- Store the vectors with the data
- Convert query to vector using Word2Vec
- Extract the text from the best result
- Build a prompt using the text as context
- Call GPT with the prompt

## Classifying data using embedding
Dataset with description, a vector representation of this description, and a category. For example, a description about a toy, its vector representation, and category ('toy'). This is similar to a classical machine learning classification problem. 

First step is loading the dataset and embedding the field that we will want to classify, in this case the description. We can work with the dataset from semantic search. 

Then, split dataset into train and test sets
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
 list(df.ada_vector.values),
 df.Category,
 test_size=0.2,
 random_seed=42
)
```

We could use different algorithms for this, such as `RandomForestClassifier`, to perform classification.

```
from sklearn.ensemble import RandomForestClassifier

# Create 100 empty trees
clf = RandomForestClassifier(n_estimators=100)

# Fit - fill all trees with the training set
clf.fit(X_train, y_train)
```

Encode and run the query:
```
# Encode the test string into an embedding vector
testme = openai.Embedding.create(
 input='<our product description>',
 model='text-embedding-ada-002'
)

# Use the trees to vote on the best value/category for our string
y_pred = clf.predict(X_test)
```

Evaluate the model
```
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Clustering data using embedding
Used to find hidden groupings within a random dataset. This is an unsupervised machine learning method. In this case, we can use GPT to cluster different reviews.

### Loading the dataset
```
import pandas as pd
import numpy as np

# read the csv file with embedded reviews
df = pd.read_csv('output/embedded_reviews.csv')

# Convert the strings stored in the ada_vector column to vectors
df['ada_vector'] = df['ada_vector'].apply(eval).apply(np.array)
```

This results into a dataset with an `ada_vector` column as a vector, and we can use the ratings for clustering.

Now, place the vectors into a list:
```
# This is our training data
vectorlist = np.vstack(df['ada_vector'].values)
```

Find the centroids using KMeans
```
from sklearn.cluster import KMeans

# Setup our KMeans parameters 
results = KMeans(
	n_clusters=4,
	init='k-means++',
	random_state=42,
	n_init='auto'
)

# Now we fit our vector list on top of the results model
results.fit(vectorlist)
```

Extract and store the clusters:
```
# results has our labels (0, 1, 2, 3 - because we set the k value to 4)
labels = results.labels_

# Create a new column in our dataframe with cluster number
df['ClusterNum'] = labels
```

Now we can explore the clusters and see what they have in common. Instead of using a loop to show entries for each cluster, let's use GPT what is common between them.
```
testcluster = 0

reviewsamples = '\n'.join(
	df[df['ClusterNum] == testcluster] # filter for cluster 0
	.combined.str.replace('Title: ', '') # removing the title
	.str.replace('\n\nContent: ', ': ') # removing the content label
	.str.replace('\n', ' ') # removing extra lines
	.sample(3, random_state=42)
	.values
)
```

We can ask the GPT model what the above reviews have in common. 

Summary:
- training dataset is kept in local machine
- do not use it to train GPT
- need to calculate vectors
- convert prompt or new entry to a vector that can be compared to local dataset

# Creative writing
 Brainstorm blog post outline ideas for the topic 'x':

 Create an outline for an artile about 'x':

What are 5 key points I should know when writing about 'topic1'?

Write 3 engaging and informative paragraphs about 'point 1 description':

'concatenaded content'
Write a short super engaging blog post introduction:

'concatenaded content'
Write a short super engaging conclusion:

We can also use GPT to expand on existing text, summarize text, extract facts from a passage, and for rewriting existing articles. 

# Creating a Chatbot with GPT
Use OpenAI `Completion` endpoint to creat a Chatbot (at least until ChatGPT API is released). 

## The Basics 
Create an interface where the chat box will be displayed using a language of choice. Then, we can call the completion function to send the question to the AI and receive a reply. 
```
