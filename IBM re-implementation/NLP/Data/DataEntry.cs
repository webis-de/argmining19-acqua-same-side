using DataAccess;
using NLP.Stemmer;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text.RegularExpressions;

namespace NLP
{
    public class DataEntry
    {
        const string testIdentifier = "test";
        readonly string[] wordSeparator = new string[] { " ", ",", ";", ":", "\t", "-", ")", "(", "\"", "[", "]", ".", "\' " };

        public int topicId;

        public string[] target;

        public bool isTest;
        public string[] claim;
        public string[] stemmedClaim;
        public string[] claimTarget;
        public int[] commaPos;
        public double claimSentiment;
        public int targetsRelation;

        private DataEntry(int topicId, string target, string test, string claimText, string claimTarget, double claimSentiment, string targetsRelation, IStemmer stemmer)
        {
            this.topicId = topicId;
            this.target = SplitSentence(target).Select(word => word.ToLower()).ToArray();
            isTest = test == testIdentifier;
            this.claimSentiment = claimSentiment;
            this.targetsRelation = int.Parse(targetsRelation);
            claim = SplitSentence(claimText).Select(word => word.ToLower()).ToArray();
            commaPos = GetCommaPos(claimText);
            stemmedClaim = claim.Select(word => stemmer.Stem(word)).ToArray();
            this.claimTarget = SplitSentence(claimTarget.ToLower());
        }

        public static DataEntry ParseRow(Row row, IStemmer stemmer)
        {
            if (!int.TryParse(row["claims.claimSentiment"], out int sentiment))
                return null;

            return new DataEntry(int.Parse(row["topicId"]),row["topicTarget"],row["split"],row["claims.claimCorrectedText"], row["claims.claimTarget.text"],sentiment,row["claims.targetsRelation"],stemmer);
        }

        public bool IsPredictionCorrect(double prediction)
        {
            var relationToTopicTarget = claimSentiment * targetsRelation;
            return relationToTopicTarget * prediction > 0;
        }

        public string[] SplitSentence(string text)
        {            
            return text.Split(wordSeparator, StringSplitOptions.RemoveEmptyEntries);
        }

        private int[] GetCommaPos(string text)
        {
            List<int> positions = new List<int>();
            foreach (Match match in Regex.Matches(text, ","))
            {
                positions.Add(match.Index);
            }
            return positions.ToArray();
        }
    }
}
