using NLP.StanceClassification.Sentiment;
using NLP.Stemmer;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NLP.StanceClassification
{
    class ClaimSentimentIdentificator
    {
        const double defaultSentiment = 0.0001;
        const int shifterScope = 8;

        readonly SentimentLexicon sentimentLexicon;
        readonly ShifterLexicon shifterLexicon;
        
        public ClaimSentimentIdentificator(IStemmer stemmer)
        {
            sentimentLexicon = new SentimentLexicon();
            shifterLexicon = new ShifterLexicon(stemmer);
        }

        public double GetTargetedSentiment(DataEntry entry, string[] target)
        {
            var targetIndices = GetTargetIndexes(entry.claim, target);

            Dictionary<int, double> sentimentForPosition = GetSentimentForWordsInClaim(entry.claim);

            ShiftSentimentIfShifterWordsPresent(entry, sentimentForPosition);

            double sentiment = GetSentimentWeightedByTargetDistance(targetIndices, sentimentForPosition);

            if (sentiment == 0)
                return defaultSentiment;

            return sentiment;
        }

        public double GetUnweightedSentiment(DataEntry entry)
        {
            Dictionary<int, double> sentimentForPosition = GetSentimentForWordsInClaim(entry.claim);

            ShiftSentimentIfShifterWordsPresent(entry, sentimentForPosition);

            double sentiment = GetUnweightedSentiment(sentimentForPosition);

            if (sentiment == 0)
                return defaultSentiment;

            return sentiment;
        }

        private Tuple<int,int> GetTargetIndexes(string[] claim, string[] claimTarget)
        {
            int targetIndex = 0;
            for (int i = 0; i < claim.Length; i++)
            {
                if (claimTarget[targetIndex] == claim[i])
                {
                    targetIndex++;
                    if (targetIndex == claimTarget.Length)
                    {
                        return Tuple.Create(i - claimTarget.Length + 1, i);
                    }
                }
                else
                {
                    targetIndex = 0;
                }
            }

            throw new ArgumentException("Target not found in claim.");
        }

        Dictionary<int, double> GetSentimentForWordsInClaim(string[] claim)
        {
            Dictionary<int, double> sentimentForPosition = new Dictionary<int, double>();

            for (int i = 0; i < claim.Length; i++)
            {
                int value = sentimentLexicon.GetSentiment(claim[i]);
                if (value != 0)
                    sentimentForPosition.Add(i, value);
            }

            return sentimentForPosition;
        }

        private void ShiftSentimentIfShifterWordsPresent(DataEntry entry, Dictionary<int, double> sentimentForPosition)
        {
            for (int i = 0; i < entry.claim.Length; i++)
            {
                if (shifterLexicon.IsRegularShifter(entry.claim[i]))
                    ApplyRegularShifter(entry.claim, entry.commaPos, sentimentForPosition, i);
                else if (shifterLexicon.IsStemmedShifter(entry.stemmedClaim[i]))
                    ApplyRegularShifter(entry.claim, entry.commaPos, sentimentForPosition, i);
                else if (shifterLexicon.IsBeforeShifter(entry.claim[i]))
                    ApplyBeforeShifter(sentimentForPosition, i);
            }
        }

        private void ApplyBeforeShifter(Dictionary<int, double> sentimentForPosition, int i)
        {
            for (int a = i - 1; a >= Math.Max(0, i - shifterScope); a--)
            {
                if (sentimentForPosition.ContainsKey(a))
                {
                    sentimentForPosition[a] *= -1;
                }
            }
        }

        private void ApplyRegularShifter(string[] claim, int[] commas, Dictionary<int, double> sentimentForPosition, int i)
        {
            for (int a = i + 1; a <= Math.Min(claim.Length-1, i + shifterScope); a++)
            {
                if ((claim[a] == "but") || commas.Contains(a))
                    return;
                if (sentimentForPosition.ContainsKey(a))
                {
                    sentimentForPosition[a] *= -1;
                }
            }
        }

        private static double GetSentimentWeightedByTargetDistance(Tuple<int,int> targetIndices, Dictionary<int, double> sentimentForPosition)
        {
            double sum = 0;
            double absSum = 0;
            foreach (var pair in sentimentForPosition)
            {
                int distance = Math.Min(Math.Abs(targetIndices.Item1 - pair.Key), Math.Abs(targetIndices.Item2 - pair.Key));
                if (pair.Key >= targetIndices.Item1 && pair.Key <= targetIndices.Item2)
                    continue;

                var weightedValue = pair.Value * Math.Pow(distance, -0.5);
                sum += weightedValue;
                absSum += Math.Abs(weightedValue);
            }

            return sum / (absSum + 1);
        }

        private static double GetUnweightedSentiment(Dictionary<int, double> sentimentForPosition)
        {
            double sentiment = 0;
            foreach (var value in sentimentForPosition)
            {
                sentiment += value.Value;
            }

            return sentiment;
        }
    }
}
