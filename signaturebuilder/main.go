package main

import (
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"sync"

	fe "github.com/kharism/cbirann/featureextract"
)

func main() {
	files, err := ioutil.ReadDir("../training")
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
	gob.Register(fe.TrainingData{})
	wg := sync.WaitGroup{}

	for _, f := range files {
		if f.IsDir() {
			wg.Add(1)
			go ExtractFeatureParallel("../training/"+f.Name(), &wg)
		}

	}
	wg.Wait()
}

func ExtractFeatureParallel(path string, wg *sync.WaitGroup) {
	pp := strings.Split(path, "/")
	defer wg.Done()
	labelName := pp[len(pp)-1]
	//fmt.Println(labelName)
	//labelFloat := encDec.EncodeLabel(labelName)
	fInfo, _ := ioutil.ReadDir(path)
	dataItem := []fe.TrainingData{}
	for _, fi := range fInfo {
		feature, _ := fe.GetFeature(path + "/" + fi.Name())
		dataItem = append(dataItem, fe.TrainingData{feature, fi.Name()})
	}
	buffer, err := os.Create(labelName + ".gob")
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
	defer buffer.Close()
	encoder := gob.NewEncoder(buffer)
	encoder.Encode(dataItem)
	fmt.Println("Done Processing", labelName)
}
