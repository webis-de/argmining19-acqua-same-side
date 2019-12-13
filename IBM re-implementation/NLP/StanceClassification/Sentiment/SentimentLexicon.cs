using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NLP.StanceClassification.Sentiment
{    class SentimentLexicon
    {
        readonly HashSet<string> positiveWords;
        readonly HashSet<string> negativeWords;

        public SentimentLexicon(string negativePath = "Resources\\negative-words.txt", string positivePath = "Resources\\positive-words.txt")
        {
            negativeWords = ParseFile(negativePath);
            positiveWords = ParseFile(positivePath);
        }

        private HashSet<string> ParseFile(string path)
        {
            HashSet<string> hashSet = new HashSet<string>();
            using (StreamReader reader = new StreamReader(path))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (!(line.StartsWith(";") || line == ""))
                    {
                        hashSet.Add(line);
                    }
                }
            }
            return hashSet;
        }

        public int GetSentiment(string word)
        {
            return positiveWords.Contains(word) ? 1 : (negativeWords.Contains(word) ? -1 : 0);
        }
    }
}
