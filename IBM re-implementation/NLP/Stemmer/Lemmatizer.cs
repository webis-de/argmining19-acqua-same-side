using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NLP.Stemmer
{
    class Lemmatizer : IStemmer
    {
        readonly LemmaSharp.Classes.Lemmatizer lemmatizer = new LemmaSharp.Classes.Lemmatizer(File.OpenRead("Resources//full7z-mlteast-en-modified.lem"));

        public override string Stem(string word)
        {
            return lemmatizer.Lemmatize(word);
        }
    }
}
