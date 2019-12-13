using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NLP.Stemmer
{
    public abstract class IStemmer
    {
        public static IStemmer GetNoStemmer() => new NoStemmer();
        public static IStemmer GetPorter2Stemmer() => new Porter2Stemmer();
        public static IStemmer GetLemmatizer() => new Lemmatizer();

        public abstract string Stem(string word);
    }

    class NoStemmer : IStemmer
    {
        public override string Stem(string word)
        {
            return word;
        }
    }
}
