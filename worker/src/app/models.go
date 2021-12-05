package main

type FindRequest struct {
	Vector     []float64 `json:"vector"`
	Neighbours []string  `json:"neighbours"`
	// number of closes vectors to return
	Quantity uint `json:"quantity"`
}

type FindResponse struct {
	Vectors []VectorRes `json:"vectors"`
}

type WriteRequest struct {
	Vector  [vecSize]float32 `json:"vector"`
	MediaID float32          `json:"mediaId"`
	Segment string           `json:"segment"`
}

type Vector struct {
	Cosine  float64   `json:"cosine"`
	Data    []float64 `json:"data"`
	MediaID float32   `json:"media_id"`
}

type VectorRes struct {
	Cosine  float64 `json:"cosine"`
	MediaID float32 `json:"media_id"`
}
