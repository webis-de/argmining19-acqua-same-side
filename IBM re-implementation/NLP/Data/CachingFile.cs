using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace NLP.Data
{
    //used for caching time-costly features (generates cache.txt in "Resources" in .exe directory, must be in the same place to be used)
    public class CachingFile
    {
        const string path = "Resources//cache.txt";
        readonly Dictionary<string, Dictionary<string, object>> dictionary;

        public CachingFile()
        {
            try
            {
                dictionary = Deserialize();
            }
            catch (Exception)
            {
                dictionary = new Dictionary<string, Dictionary<string, object>>();
            }
        }

        public void Add(string method, string key, object value)
        {
            if (!dictionary.ContainsKey(method))
                dictionary.Add(method, new Dictionary<string, object>());

            dictionary[method].Add(key, value);
        }

        public T Get<T>(string method, string key)
        {
            if (dictionary.ContainsKey(method))
            {
                if (dictionary[method].ContainsKey(key))
                {
                    return (T)dictionary[method][key];
                }
            }
            return default;
        }

        public void SaveToFile()
        {
            Serialize(dictionary);
        }


        private void Serialize(Dictionary<string, Dictionary<string, object>> dictionary)
        {
            FileStream stream = new FileStream(path, FileMode.Create);

            using (stream)
            {
                BinaryFormatter bin = new BinaryFormatter();
                bin.Serialize(stream, dictionary);
            }
        }

        private Dictionary<string, Dictionary<string, object>> Deserialize()
        {
            FileStream stream = new FileStream(path, FileMode.Open);

            using (stream)
            {
                BinaryFormatter bin = new BinaryFormatter();
                return (Dictionary<string, Dictionary<string, object>>)bin.Deserialize(stream);
            }
        }
    }
}
