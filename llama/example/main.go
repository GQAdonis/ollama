package main

import (
	"flag"
	"fmt"

	"github.com/ollama/ollama/llama"
)

func main() {
	mp := flag.String("model", "", "Path to model binary file")
	prompt := flag.String("prompt", "", "String prompt for generation")
	limit := flag.Int("limit", 100, "Number of tokens to predict")
	flag.Parse()

	if mp == nil {
		fmt.Println("No model specified")
		return
	}

	llama.BackendInit()
	params := llama.NewModelParams()
	model := llama.LoadModelFromFile(*mp, params)
	ctxParams := llama.NewContextParams()
	lc := llama.NewContextWithModel(model, ctxParams)
	if lc == nil {
		panic("Failed to create context")
	}

	tokens, err := model.Tokenize(*prompt, 2048, true, true)
	if err != nil {
		panic(err)
	}

	batch := llama.NewBatch(512, 0, 1)

	// prompt eval
	for i, t := range tokens {
		batch.Add(t, llama.Pos(i), []llama.SeqId{0}, true)
	}

	// main loop
	for n := batch.NumTokens(); n < *limit; n++ {
		err = lc.Decode(batch)
		if err != nil {
			panic("Failed to decode")
		}

		// sample a token
		token := lc.SampleTokenGreedy(batch)

		// if it's an end of sequence token, break
		if model.TokenIsEog(token) {
			break
		}

		// print the token
		str := model.TokenToPiece(token)
		fmt.Print(str)

		batch.Clear()
		batch.Add(token, llama.Pos(n), []llama.SeqId{0}, true)
	}
}
