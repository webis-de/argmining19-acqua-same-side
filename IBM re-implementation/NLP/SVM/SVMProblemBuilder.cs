using libsvm;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NLP
{
    public class SVMProblemBuilder
    {
        public svm_problem CreateProblem(List<DataEntry> dataEntry, List<string> vocabulary)
        {
            double[] y = new double[dataEntry.Count];
            svm_node[][] x = new svm_node[dataEntry.Count][];

            for (int i = 0; i < dataEntry.Count; i++)
            {
                y[i] = dataEntry[i].claimSentiment;
                x[i] = CreateNode(dataEntry[i].stemmedClaim, vocabulary);
            }

            return new svm_problem
            {
                y = y,
                x = x,
                l = y.Length
            };
        }

        public static svm_node[] CreateNode(string[] words, List<string> vocabulary)
        {
            var node = new List<svm_node>(vocabulary.Count);

            for (int i = 0; i < vocabulary.Count; i++)
            {
                int occurenceCount = words.Count(s => String.Equals(s, vocabulary[i], StringComparison.OrdinalIgnoreCase));
                if (occurenceCount == 0)
                    continue;

                node.Add(new svm_node
                {
                    index = i + 1,
                    value = occurenceCount
                });
            }

            return node.ToArray();
        }
    }
}