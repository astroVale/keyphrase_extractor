# keywords_generator
Python 3 package which generates the most important key-phrases / key-words from text files.

## Scripts:
* The *scripts.zip* archive contains:
   - one script file (*script.txt*)
   - 3 transcript files (*transcript1...3.txt*)
* ```keyphrases_extraction.py``` - Python class to extract key-phrases from text files
* ```main.py``` - The main script to run and get key-phrases
*  ```keywords_extraction.ipynb``` - Jupyter Notebook for keywords extraction
 
## Instructions
1. Install the necessary requirements using the ```requirements.txt``` file.
2. Unzip the archive
3. Run the python script

## TO DO
1. Compute the most important keywords (a keyword can be between 1-3 words)
2. Choose the top n words from the previously generated list. Compare these keywords with all the words occurring in all of the transcripts.
3. Generate a score (rank) for these top n words based on analysed transcripts.
