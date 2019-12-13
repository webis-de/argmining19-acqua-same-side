using edu.stanford.nlp.trees;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NLP.StanceClassification.Target
{
    public class NounPhrase
    {
        public string[] PhraseArray { get; private set; }
        public string PhraseString { get; private set; }

        public string[] TargetArray { get; private set; }
        public string TargetString { get; private set; }
        public string[] Sentence { get; private set; }
        public Tree Tree { get; private set; }
        public java.util.List Dependencies { get; private set; }

        private NounPhrase(string[] sentence, Tree tree, java.util.List dependencies, string[] target)
        {
            TargetArray = target;
            TargetString = string.Join(" ", target);
            Sentence = sentence;
            Dependencies = dependencies;
            Tree = tree;
        }


        public static NounPhrase SetSentence(string[] sentence, Tree tree, java.util.List dependencies, string[] target)
        {
            return new NounPhrase(sentence, tree, dependencies, target);
        }

        public void SetPhrase(string phrase)
        {
            PhraseString = phrase;
            PhraseArray = phrase.Split(' ');
        }
    }
}
