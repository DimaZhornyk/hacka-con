package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/labstack/echo/v4"
)

const (
	dataPath = "data/"
	vecSize  = 1024
)

func main() {
	e := echo.New()

	initRoutes(e)

	log.Fatal(e.Start(fmt.Sprintf(":%d", 9000)))
}

func initRoutes(e *echo.Echo) {
	e.POST("/find", handler)
	e.POST("/write", writer)
}

func writer(ctx echo.Context) error {
	req := new(WriteRequest)
	if err := ctx.Bind(req); err != nil {
		return ctx.JSON(http.StatusBadRequest, "Error binding a request")
	}
	f, err := os.OpenFile(dataPath+req.Segment, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0777)
	if err != nil {
		log.Println(err.Error())
		return ctx.JSON(http.StatusInternalServerError, map[string]string{"error": "Error while opening a file"})
	}
	defer f.Close()

	b := make([]byte, 0)
	b = append(b, float32ToBytes(req.MediaID)...)
	for _, n := range req.Vector {
		b = append(b, float32ToBytes(n)...)
	}

	if _, err := f.Write(b); err != nil {
		log.Println(err.Error())
		return ctx.JSON(http.StatusInternalServerError, map[string]string{"error": "Error Writing to a file"})
	}

	return ctx.NoContent(http.StatusNoContent)
}

func handler(ctx echo.Context) error {
	req := new(FindRequest)
	if err := ctx.Bind(req); err != nil {
		return ctx.JSON(http.StatusBadRequest, "Error binding a request")
	}

	wg := &sync.WaitGroup{}
	vectors := make(chan []Vector)
	for _, n := range req.Neighbours {
		wg.Add(1)
		go func(neighbour string, best chan []Vector) {
			defer wg.Done()
			// map of cosine similarity to vector
			bestN := make([]Vector, req.Quantity)

			// open the neighbour and extract it's data
			path := dataPath + neighbour
			f, err := os.OpenFile(path, os.O_RDONLY|os.O_CREATE, 0777)
			if err != nil {
				if os.IsNotExist(err) {
					return
				}
				log.Println(err.Error())
				return
			}
			defer f.Close()

			// read from by vector and count its cosine
			for {
				buf := make([]byte, vecSize*4+4)
				if _, err := f.Read(buf); err != nil {
					if err == io.EOF {
						break
					}
				}

				mediaId := bytesToFloat32(buf[0:4])
				vec := make([]float64, 0)
				for i := 4; i < vecSize*4; i += 4 {
					vec = append(vec, float64(bytesToFloat32(buf[i:i+4])))
				}

				similarity, err := cosine(req.Vector, vec)
				if err != nil {
					log.Println(err.Error())
					return
				}
				// if current is higher than minimal, rewrite it
				minIdx := findMinIndex(&bestN)
				if similarity > bestN[minIdx].Cosine {
					bestN[minIdx] = Vector{Cosine: similarity, Data: vec, MediaID: mediaId}
				}
			}
			best <- bestN
		}(n, vectors)
	}

	res := make([]Vector, req.Quantity)
	done := make(chan struct{})

	go func() {
		bestN := make([]Vector, req.Quantity)
		for {
			select {
			case batch := <-vectors:
				for _, v := range batch {
					minIdx := findMinIndex(&bestN)
					if v.Cosine > bestN[minIdx].Cosine {
						bestN[minIdx] = v
					}
				}
			case <-done:
				res = bestN
				return
			}
		}
	}()

	wg.Wait()
	done <- struct{}{}

	r := make([]VectorRes, 0)
	for i := range res {
		r = append(r, VectorRes{Cosine: res[i].Cosine, MediaID: res[i].MediaID})
	}

	return ctx.JSON(http.StatusOK, FindResponse{Vectors: r})
}
