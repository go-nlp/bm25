package bm25

import (
	"fmt"
	"sort"
	"strings"

	"github.com/go-nlp/tfidf"
)

var mobydick = []string{
	"Call me Ishmael .",
	"Some years ago -- never mind how long precisely -- having little or no money in my purse , and nothing particular to interest me on shore , I thought I would sail about a little and see the watery part of the world .",
	"It is a way I have of driving off the spleen and regulating the circulation .",
	"Whenever I find myself growing grim about the mouth ; ",
	"whenever it is a damp , drizzly November in my soul ; ",
	"whenever I find myself involuntarily pausing before coffin warehouses , and bringing up the rear of every funeral I meet ; ",
	"and especially whenever my hypos get such an upper hand of me , that it requires a strong moral principle to prevent me from deliberately stepping into the street , and methodically knocking people's hats off -- then , I account it high time to get to sea as soon as I can .",
	"This is my substitute for pistol and ball . ",
	"With a philosophical flourish Cato throws himself upon his sword ; ",
	"I quietly take to the ship . There is nothing surprising in this .",
	"If they but knew it , almost all men in their degree , some time or other , cherish very nearly the same feelings towards the ocean with me .",
}

type doc []int

func (d doc) IDs() []int { return []int(d) }

func makeCorpus(a []string) (map[string]int, []string) {
	retVal := make(map[string]int)
	invRetVal := make([]string, 0)
	var id int
	for _, s := range a {
		for _, f := range strings.Fields(s) {
			f = strings.ToLower(f)
			if _, ok := retVal[f]; !ok {
				retVal[f] = id
				invRetVal = append(invRetVal, f)
				id++
			}
		}
	}
	return retVal, invRetVal
}

func makeDocuments(a []string, c map[string]int) []tfidf.Document {
	retVal := make([]tfidf.Document, 0, len(a))
	for _, s := range a {
		var ts []int
		for _, f := range strings.Fields(s) {
			f = strings.ToLower(f)
			id := c[f]
			ts = append(ts, id)
		}
		retVal = append(retVal, doc(ts))
	}
	return retVal
}

func Example_BM25() {
	corpus, _ := makeCorpus(mobydick)
	docs := makeDocuments(mobydick, corpus)
	tf := tfidf.New()

	for _, doc := range docs {
		tf.Add(doc)
	}
	tf.CalculateIDF()

	// now we search

	// "ishmael" is a query
	ishmael := doc{corpus["ishmael"]}

	// "whenever i find" is another query
	whenever := doc{corpus["whenever"]}

	ishmaelScores := BM25(tf, ishmael, docs, 1.5, 0.75)
	wheneverScores := BM25(tf, whenever, docs, 1.5, 0.75)

	sort.Sort(sort.Reverse(ishmaelScores))
	sort.Sort(sort.Reverse(wheneverScores))

	fmt.Printf("Top 3 Relevant Docs to \"Ishmael\":\n")
	for _, d := range ishmaelScores[:3] {
		fmt.Printf("\tID   : %d\n\tScore: %1.3f\n\tDoc  : %q\n", d.ID, d.Score, mobydick[d.ID])
	}
	fmt.Println("")
	fmt.Printf("Top 3 Relevant Docs to \"whenever i find\":\n")
	for _, d := range wheneverScores[:3] {
		fmt.Printf("\tID   : %d\n\tScore: %1.3f\n\tDoc  : %q\n", d.ID, d.Score, mobydick[d.ID])
	}
	// Output:
	// Top 3 Relevant Docs to "Ishmael":
	//	ID   : 0
	//	Score: 3.706
	//	Doc  : "Call me Ishmael ."
	//	ID   : 1
	//	Score: 0.000
	//	Doc  : "Some years ago -- never mind how long precisely -- having little or no money in my purse , and nothing particular to interest me on shore , I thought I would sail about a little and see the watery part of the world ."
	//	ID   : 2
	//	Score: 0.000
	//	Doc  : "It is a way I have of driving off the spleen and regulating the circulation ."
	//
	// Top 3 Relevant Docs to "whenever i find":
	//	ID   : 3
	//	Score: 2.031
	//	Doc  : "Whenever I find myself growing grim about the mouth ; "
	//	ID   : 4
	//	Score: 1.982
	//	Doc  : "whenever it is a damp , drizzly November in my soul ; "
	//	ID   : 5
	//	Score: 1.810
	//	Doc  : "whenever I find myself involuntarily pausing before coffin warehouses , and bringing up the rear of every funeral I meet ; "

}
