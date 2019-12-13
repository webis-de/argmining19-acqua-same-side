using Porter2StemmerStandard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NLP.Stemmer
{
    class Porter2Stemmer : IStemmer
    {
        readonly EnglishPorter2Stemmer stemmer = new EnglishPorter2Stemmer();

        public override string Stem(string word)
        {
            return stemmer.Stem(word).Value;
        }
    }
}
