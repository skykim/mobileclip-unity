using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
public class BPETokenizer
{
    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<int, string> _decoder;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<string, List<string>> _bpeCache = new Dictionary<string, List<string>>();
    
    private readonly Dictionary<byte, char> _byteEncoder;
    private readonly Dictionary<char, byte> _byteDecoder;

    private readonly Regex _tokenPatternRegex;

    public int BosTokenId { get; }
    public int EosTokenId { get; }
    private const string TokenizePattern = @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

    public BPETokenizer(string tokenizerJsonContent)
    {
        JObject root = JObject.Parse(tokenizerJsonContent);
        JObject model = root["model"] as JObject;

        if (model == null)
        {
            throw new Exception("Invalid tokenizer.json format: 'model' field missing.");
        }

        _encoder = model["vocab"].ToObject<Dictionary<string, int>>();
        _decoder = _encoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        var mergesList = model["merges"].ToObject<List<string>>();
        _bpeRanks = LoadMergesFromList(mergesList);

        (_byteEncoder, _byteDecoder) = BuildByteToUnicodeMap();

        _tokenPatternRegex = new Regex(TokenizePattern, RegexOptions.Compiled | RegexOptions.IgnoreCase);

        if (_encoder.TryGetValue("<|startoftext|>", out int bosId)) BosTokenId = bosId;
        else BosTokenId = 49406;

        if (_encoder.TryGetValue("<|endoftext|>", out int eosId)) EosTokenId = eosId;
        else EosTokenId = 49407;
    }

    public int[] Encode(string text, int contextLength = 77)
    {
        text = text.Replace("\n", " ").Replace("\r", " ");
        text = Regex.Replace(text, @"\s+", " ").Trim().ToLower();

        List<int> tokens = new List<int> { BosTokenId };

        MatchCollection matches = _tokenPatternRegex.Matches(text);
        foreach (Match match in matches)
        {
            string token = match.Value;
            
            StringBuilder sb = new StringBuilder();
            byte[] bytes = Encoding.UTF8.GetBytes(token);
            foreach (byte b in bytes)
            {
                sb.Append(_byteEncoder[b]);
            }
            string encodedToken = sb.ToString();

            List<string> bpeTokens = Bpe(encodedToken);
            
            foreach (string bpeToken in bpeTokens)
            {
                if (_encoder.TryGetValue(bpeToken, out int id))
                {
                    tokens.Add(id);
                }
            }
        }

        tokens.Add(EosTokenId);

        if (tokens.Count > contextLength)
        {
            tokens = tokens.Take(contextLength).ToList();
            tokens[contextLength - 1] = EosTokenId;
        }
        else
        {
            while (tokens.Count < contextLength)
            {
                tokens.Add(0);
            }
        }

        return tokens.ToArray();
    }

    private List<string> Bpe(string token)
    {
        if (_bpeCache.TryGetValue(token, out var cached)) return cached;

        List<string> word = token.Select(c => c.ToString()).ToList();
        int lastIndex = word.Count - 1;
        word[lastIndex] = word[lastIndex] + "</w>";

        while (word.Count > 1)
        {
            var pairs = GetPairs(word);
            if (pairs.Count == 0) break;

            var bigram = pairs.OrderBy(p => _bpeRanks.TryGetValue(p, out int rank) ? rank : int.MaxValue).First();

            if (!_bpeRanks.ContainsKey(bigram)) break;

            List<string> newWord = new List<string>();
            int i = 0;
            while (i < word.Count)
            {
                if (i < word.Count - 1 && word[i] == bigram.Item1 && word[i+1] == bigram.Item2)
                {
                    newWord.Add(bigram.Item1 + bigram.Item2);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i++;
                }
            }
            word = newWord;
        }

        _bpeCache[token] = word;
        return word;
    }

    private HashSet<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new HashSet<(string, string)>();
        for (int i = 0; i < word.Count - 1; i++)
        {
            pairs.Add((word[i], word[i + 1]));
        }
        return pairs;
    }

    private Dictionary<(string, string), int> LoadMergesFromList(List<string> mergesList)
    {
        var ranks = new Dictionary<(string, string), int>();
        int rank = 0;

        foreach (string line in mergesList)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int lastSpace = line.LastIndexOf(' ');
            if (lastSpace == -1) continue;

            string part1 = line.Substring(0, lastSpace);
            string part2 = line.Substring(lastSpace + 1);
            
            ranks[(part1, part2)] = rank++;
        }
        return ranks;
    }

    private (Dictionary<byte, char>, Dictionary<char, byte>) BuildByteToUnicodeMap()
    {
        var bs = new List<byte>();
        for (int i = 33; i <= 126; i++) bs.Add((byte)i);
        for (int i = 161; i <= 172; i++) bs.Add((byte)i);
        for (int i = 174; i <= 255; i++) bs.Add((byte)i);

        var cs = new List<char>();
        int n = 0;
        
        var bsSet = new HashSet<byte>(bs);
        var mapping = new Dictionary<byte, char>();
        var inverseMapping = new Dictionary<char, byte>();

        for (int i = 0; i < 256; i++)
        {
            byte b = (byte)i;
            char c = bsSet.Contains(b) ? (char)b : (char)(256 + n++);
            mapping[b] = c;
            inverseMapping[c] = b;
        }
        return (mapping, inverseMapping);
    }
}