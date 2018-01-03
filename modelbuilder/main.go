package main

import (
	"encoding/gob"
	"fmt"

	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"sync"

	fe "github.com/kharism/cbirann/featureextract"
	"github.com/kharism/gobrain"
)

type EncoderDecoder interface {
	EncodeLabel(label string) []float64
	DecodeLabel(label []float64) string
}

type BasicEncoderDecoder struct {
	labels []string
}

func (c BasicEncoderDecoder) EncodeLabel(label string) []float64 {
	hasil := []float64{}
	for _, l := range c.labels {
		if l == label {
			hasil = append(hasil, 1.0)
		} else {
			hasil = append(hasil, 0.0)
		}
	}
	return hasil
}
func (c BasicEncoderDecoder) DecodeLabel(label []float64) string {
	maxIndex := 0
	maxValue := 0.0
	for i, l := range label {
		if l > maxValue {
			maxIndex = i
			maxValue = l
		}
	}
	return c.labels[maxIndex]
}

var trainingData [][][]float64
var testingData [][][]float64

func main() {
	rand.Seed(1)
	files, err := ioutil.ReadDir("../training")
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
	files2, err := ioutil.ReadDir("../testing")
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
	// f, err := os.Create("cpu.pprof")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// pprof.StartCPUProfile(f)
	// defer pprof.StopCPUProfile()
	labelName := []string{}
	for _, f := range files {
		labelName = append(labelName, f.Name())
	}
	labelName2 := []string{}
	for _, f := range files2 {
		labelName2 = append(labelName2, f.Name())
	}
	//fmt.Println(labelName)
	gob.Register(gobrain.FeedForward{})
	//build training data
	ann := gobrain.FeedForward{}
	encDec := BasicEncoderDecoder{labelName}
	wg := sync.WaitGroup{}
	cummulator := make(chan [][][]float64, len(labelName)+1)
	for _, label := range labelName {
		wg.Add(1)
		fmt.Println("Processing", label)
		go ExtractFeatureParallel("../training/"+label, encDec, cummulator, &wg)
	}
	wg.Wait()
	close(cummulator)
	for d := range cummulator {
		fmt.Println(len(d))
		trainingData = append(trainingData, d...)
	}

	wg2 := sync.WaitGroup{}
	cummulator2 := make(chan [][][]float64, len(labelName)+1)
	for _, label := range labelName2 {
		wg2.Add(1)
		fmt.Println("Processing", label)
		go ExtractFeatureParallel("../testing/"+label, encDec, cummulator2, &wg2)
	}
	wg2.Wait()
	close(cummulator2)
	for d := range cummulator2 {
		fmt.Println(len(d))
		testingData = append(testingData, d...)
	}

	if _, err := os.Stat("model.gob"); os.IsNotExist(err) {

		//fmt.Println(trainingData)
		inputNum := len(trainingData[0][0])
		outputNum := len(trainingData[0][1])
		fmt.Println("Input", inputNum)
		fmt.Println("Output", outputNum)
		ann.Init(inputNum, inputNum/20, outputNum)
		bias := [][]float64{}
		bias = append(bias, []float64{})
		for i := 0; i <= (inputNum / 20); i++ {
			bias[0] = append(bias[0], rand.Float64())
		}

		//ann.SetContexts(0, bias)
		//ann.Worker = 4
		fmt.Println("START TRAINING")
		ann.Train(trainingData, 400, 0.1, 0.1, true)
		buffer, err := os.Create("model.gob")
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
		defer buffer.Close()
		enc := gob.NewEncoder(buffer)
		err = enc.Encode(ann)
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
		fmt.Println("DONE TRAINING")
	} else {

		buffer, err := os.Open("model.gob")
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
		defer buffer.Close()
		dec := gob.NewDecoder(buffer)
		err = dec.Decode(&ann)
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
	}
	fmt.Println("DONE LOADING MODEL")

	for _, data := range trainingData {
		//fmt.Println(data[0])
		results := ann.Update(data[0])
		fmt.Println(encDec.DecodeLabel(results), encDec.DecodeLabel(data[1]))
	}
	fmt.Println("test testing data")
	fmt.Println("========")
	for _, data := range testingData {
		results := ann.Update(data[0])
		resStr := encDec.DecodeLabel(results)
		resStr2 := encDec.DecodeLabel(data[1])
		fmt.Println(resStr, resStr2)
		if resStr != resStr2 {
			fmt.Println(results, data[1])
		}
	}

}
func ExtractFeatureParallel(path string, encDec EncoderDecoder, cummulator chan [][][]float64, wg *sync.WaitGroup) {
	pp := strings.Split(path, "/")
	defer wg.Done()
	labelName := pp[len(pp)-1]
	//fmt.Println(labelName)
	labelFloat := encDec.EncodeLabel(labelName)
	fInfo, _ := ioutil.ReadDir(path)
	dataItem := [][][]float64{}
	for _, fi := range fInfo {
		feature := fe.GetFeature(path + "/" + fi.Name())
		dataItem = append(dataItem, [][]float64{feature, labelFloat})
	}
	//fmt.Println("PAR", dataItem)
	cummulator <- dataItem
	fmt.Println("Done Processing", labelName)
}
