using System;
using System.Collections.Generic;

[Serializable]
public class EmbeddingData
{
    public string imageName;
    public float[] embedding;
}

[Serializable]
public class EmbeddingIndex
{
    public List<EmbeddingData> indexList;
}

[Serializable]
public struct SimilarityResult
{
    public string imageName;
    public float score;
}