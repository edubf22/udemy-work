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
 

