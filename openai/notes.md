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
<code for a broken function>
"""
Explain why the previous function doesn't work.
"""

## Converting between languages
```
# Convert this from Python to R
# Python Version

[ Python code ]

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
"Summary: <summary of the interaction so far>\n\nSpecific information: <e.g. order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\n\nCustomer: <message2>\nAgent:"
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
