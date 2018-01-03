package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/kharism/cbirann/featureextract"

	"github.com/eaciit/config"
	"github.com/kharism/gobrain"
)

func DefaultHandler(w http.ResponseWriter, r *http.Request) {
	b, err := ioutil.ReadFile("default.html")
	if err != nil {
		fmt.Println(err.Error())
	}
	w.Write(b)
	/*templates, err := template.New("default.html").ParseFiles("default.html")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	err = templates.Execute(w, nil)*/

}
func AssetHandler(w http.ResponseWriter, r *http.Request) {
	//fmt.Println(r.URL.Path)
	//f := strings.Split(r.URL.Path, "/")
	filename := "./" + r.URL.Path
	b, _ := ioutil.ReadFile(filename)
	w.Write(b)
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
func UploadHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("method:", r.Method)
	if r.Method == "POST" {
		r.ParseMultipartForm(32 << 20)
		file, _, err := r.FormFile("target")
		if err != nil {
			fmt.Println(err)
			return
		}

		defer file.Close()
		//fmt.Fprintf(w, "%v", handler.Header)
		features, _ := featureextract.GetFeatureFile(file)
		results := ann.Update(features)
		hasil := struct {
			Label   string
			Similar []string
		}{}
		hasil.Label = encDec.DecodeLabel(results)

		db := dbMap[hasil.Label]
		nn := featureextract.FindKNN(features, 3, db)
		for _, n := range nn {
			hasil.Similar = append(hasil.Similar, n.Filename)
		}
		r, _ := json.Marshal(hasil)
		w.Write(r)
	} else {
		w.Write([]byte("NO GET Allowed"))
	}
}
func GenerateAssetFunc(handler, path string) func(w http.ResponseWriter, r *http.Request) {
	p := func(w http.ResponseWriter, r *http.Request) {
		filename := path + "/" + strings.Replace(r.URL.Path, handler, "", 1)
		b, err := ioutil.ReadFile(filename)
		if err == nil {
			if strings.HasSuffix(filename, "css") {
				w.Header().Set("Content-type", "text/css")
			}
			if strings.HasSuffix(filename, "js") {
				w.Header().Set("Content-type", "application/javascript")
			}
			w.Write(b)
		} else {
			http.Error(w, err.Error(), 404)
		}

	}
	return p
}

var encDec BasicEncoderDecoder
var ann gobrain.FeedForward
var dbMap map[string][]featureextract.TrainingData

func main() {
	if e := config.SetConfigFile("config.json"); e != nil {
		panic(e.Error())
	}
	dbMap = map[string][]featureextract.TrainingData{}
	port := config.Get("Port").(string)
	http.HandleFunc("/assets/", GenerateAssetFunc("/assets/", "./assets"))
	http.HandleFunc("/", DefaultHandler)
	http.HandleFunc("/training/", GenerateAssetFunc("/training/", "../training"))
	http.HandleFunc("/upload/", UploadHandler)

	files, err := ioutil.ReadDir("../training")
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
	labelName := []string{}
	for _, f := range files {
		dd, _ := os.Open(f.Name() + ".gob")
		dec := gob.NewDecoder(dd)
		bb := []featureextract.TrainingData{}
		err := dec.Decode(&bb)
		dd.Close()
		dbMap[f.Name()] = bb
		fmt.Println(f.Name(), ">>", len(bb))
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
		labelName = append(labelName, f.Name())
	}

	encDec = BasicEncoderDecoder{labelName}
	if _, err := os.Stat("model.gob"); err == nil {
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
	} else {
		os.Exit(1)
	}
	//http.Handle("/assets/js", http.FileServer(http.Dir("assets/js")))
	//http.Handle("/assets/font", http.FileServer(http.Dir("assets/font")))
	//http.Handle("/training", http.FileServer(http.Dir("../training")))

	fmt.Println("Start Serving", ":"+port)
	//strconv.FormatFloat(port,"d",)
	http.ListenAndServe(":"+port, nil)
}
