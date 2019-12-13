using NLP.Stemmer;
using System.Collections.Generic;

namespace NLP.StanceClassification.Sentiment
{
    class ShifterLexicon
    {
        readonly HashSet<string> regularShifters;
        readonly HashSet<string> stemmedShifters;
        readonly HashSet<string> beforeShifters;

        public ShifterLexicon(IStemmer stemmer)
        {
            regularShifters = new HashSet<string>(new string[] {
            "non","no","neither","nor","not","don't","cannot","doesn't","can't",
            "shouldn't","wouldn't","isn't","aren't","n't",
            "haven't","hasn't", "never","against","anti","without",
            "little","hardly","barely","fewer","smaller","although","despite","less", "least"
            });

            var stemmedShifterOrigin = new string[]
            {
            "abate", "alleviate", "amputate", "atrophy", "cheapen", "collapse",
            "contract", "corrode", "corrosion", "corrosive", "counteract",
            "counteraction", "cut", "decay", "decline", "decompose", "decrease",
            "deplete", "depreciate", "depreciation", "detract", "dim", "diminish",
            "discount", "discouragement", "dispel", "dispense", "drain", "dwindle",
            "ease", "empty", "engulf", "eradicate", "erase", "erode", "erosion",
            "exasperate", "exhaust", "exterminate", "extermination",
            "falter", "languish", "leakage", "lighten", "lower", "low","melt", "limit",
            "minimize", "pass", "purify", "ration", "recede", "recession", "reduce",
            "reduction", "refine", "retardation", "reverse", "rid", "rot", "scarcity",
            "shrank", "shred", "shrink", "shrivel", "shrunk", "slow", "subtract",
            "sunder", "tatter", "vanish", "wane", "weaken", "wilt", "restrict","restriction",
            "wither", "lack", "curtail", "oppose", "devalue", "undermine", "interfere", "avoid", "prevent", "stop","contradict",
            "hinder","threaten","deny","denies","respond", "solve","violate","deter" ,"deterrent", "difficult","impossible"
            };

            stemmedShifters = new HashSet<string>();
            foreach (var shifter in stemmedShifterOrigin)
            {
                stemmedShifters.Add(shifter);
                stemmedShifters.Add(stemmer.Stem(shifter));
            }

            beforeShifters = new HashSet<string>(new string[]
            {
            "rare", "uncommon","unlikely","impossible",
            "decreased","reduced","miminized","alleviated",
            "abated",  "amputated", "atrophied", "cheapen", "collapsed",
            "contracedt", "corroded" ,"cut", "decayed", "declined", "decomposed", "decreased",
            "depleted", "depreciated", "detracted", "dimmed", "diminished",
            "discounted", "dispensed", "drained", "dwindled",
            "eased", "emptied","eradicated", "erased", "eroded",
            "exasperate", "exhausted", "exterminated", "faded",
            "faltered", "languished", "leaked", "lightened", "lowered", "melted",
            "minimized","receded", "reduced", "restricted",
            "reversed", "shrank", "shreded", "shriveled", "shrunk", "slowed", "subsided", "subtracted",
            "sundered", "tattered", "vanihed", "waned", "weaken", "wilted",  "withered",
            "devalued","limited", "undermined", "interfered", "avoided", "prevented", "stopped","contradicted",
            "threatened","denied","opposed","cured","violated","lacking"
            });
        }


        public bool IsRegularShifter(string word)
        {
            return regularShifters.Contains(word);
        }

        public bool IsStemmedShifter(string word)
        {
            return stemmedShifters.Contains(word);
        }

        public bool IsBeforeShifter(string word)
        {
            return beforeShifters.Contains(word);
        }
    }
}
