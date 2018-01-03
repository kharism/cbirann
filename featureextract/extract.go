package featureextract

import (
	"errors"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"sort"
	"sync"

	colorful "github.com/lucasb-eyer/go-colorful"
)

type TrainingData struct {
	Signature []float64
	Filename  string
}

type rgba struct {
	R, G, B float64
}

func (a *rgba) Add(R, G, B float64) {
	a.R += R //uint64(R)
	a.G += G //uint64(G)
	a.B += B //uint64(B)
	//a.A += uint64(A * A)
}
func (a *rgba) Divide(A float64) {
	a.R /= A //uint64()
	a.G /= A //uint64(A)
	a.B /= A //uint64(A)
	//a.A /= uint64(A)
}
func (a *rgba) Normalize() {
	a.R = a.R * 255 / 65535
	a.G = a.G * 255 / 65535
	a.B = a.B * 255 / 65535
	//a.A = a.A * 255 / 65535
}

func (rgb *rgba) ToArray() []float64 {
	//c := colorful.Color{math.Sqrt(rgb.R), math.Sqrt(rgb.G), math.Sqrt(rgb.B)}
	//l, a, b := c.Lab()
	hasil := []float64{rgb.R, rgb.G, rgb.B}
	return hasil
}

type colorMap [][]colorful.Color

func (a colorMap) ToArray() []colorful.Color {
	p := []colorful.Color{}
	for _, v := range a {
		for _, c := range v {
			p = append(p, c)
		}
	}
	return p
}
func (a colorMap) ToColorArray() []colorful.Color {
	p := []colorful.Color{}
	for _, v := range a {
		for _, c := range v {
			p = append(p, c)
		}
	}
	return p
}
func (a colorMap) ToFloatArray() []float64 {
	p := []float64{}
	for _, v := range a {
		for _, c := range v {
			r, g, b := c.FastLinearRgb()
			p = append(p, r, g, b)
		}
	}
	return p
}

const BLOCK_SIZE = 15

func GetFeatureFile(r io.Reader) ([]float64, []colorful.Color) {
	img, _, _ := image.Decode(r)

	size := img.Bounds()
	imgRGBA := image.NewRGBA(size)
	draw.Draw(imgRGBA, img.Bounds(), img, image.ZP, draw.Src)
	LayoutColors := make([][]colorful.Color, BLOCK_SIZE)
	for i := 0; i < BLOCK_SIZE; i++ {
		LayoutColors[i] = make([]colorful.Color, BLOCK_SIZE)
	}
	for i := 0; i < BLOCK_SIZE; i++ {
		for j := 0; j < BLOCK_SIZE; j++ {
			bounds := image.Rect(i*int(math.Ceil(float64(size.Dx())/BLOCK_SIZE)), j*int(math.Ceil(float64(size.Dy())/BLOCK_SIZE)), (i+1)*int(math.Ceil(float64(size.Dx())/BLOCK_SIZE)), (j+1)*int(math.Ceil(float64(size.Dy())/BLOCK_SIZE)))

			subImage := imgRGBA.SubImage(bounds)
			subBounds := subImage.Bounds()
			newRGBA := rgba{}
			count := 0
			for i1 := subBounds.Min.X; i1 < subBounds.Max.X; i1++ {
				for j1 := subBounds.Min.Y; j1 < subBounds.Max.Y; j1++ {
					count += 1
					c := colorful.MakeColor(subImage.At(i1, j1))

					newRGBA.Add(c.R*c.R, c.G*c.G, c.B*c.B)
				}
			}
			newRGBA.Divide(float64(count))
			//newRGBA.Normalize()
			if count > 0 {
				LayoutColors[i][j] = colorful.Color{newRGBA.R, newRGBA.G, newRGBA.B}
			}

		}
	}
	l := colorMap(LayoutColors)
	return l.ToFloatArray(), l.ToColorArray()
}
func GetFeature(filename string) ([]float64, []colorful.Color) {
	r, _ := os.Open(filename)
	defer r.Close()
	return GetFeatureFile(r)
}

type bestResult struct {
	sync.RWMutex
	best []SortableStruct
	max  int
}

func (b *bestResult) Add(s SortableStruct) {
	if len(b.best) < b.max {
		b.best = append(b.best, s)
	} else {
		if b.best[b.max-1].distance > s.distance {
			b.best = b.best[:b.max-1]
			b.best = append(b.best, s)
		}
	}
	sort.Slice(b.best, func(i, j int) bool {
		return b.best[i].distance < b.best[j].distance
	})
}
func sq(v float64) float64 {
	return v * v
}
func DistanceLab(c1, c2 colorful.Color) (float64, error) {
	l1, a1, b1 := c1.R, c1.G, c1.B
	l2, a2, b2 := c2.R, c2.G, c2.B
	hasil := sq(l1-l2) + sq(a1-a2) + sq(b1-b2)
	if math.IsNaN(hasil) {
		fmt.Println(l1, a1, b1, l2, a2, b2)
		return 0, errors.New("Nan Error")
	}
	return hasil, nil
}
func Float64Dist(a, b []float64) float64 {
	dist := 0.0

	for i := 0; i < len(a); i++ {
		//fmt.Println(a[i].R, a[i].G, a[i].B)
		dist += (a[i] - b[i]) * (a[i] - b[i])
	}
	//fmt.Println(dist)
	return dist
}
func HclDist(a, b []colorful.Color) float64 {
	dist := 0.0

	for i := 0; i < len(a); i++ {
		//fmt.Println(a[i].R, a[i].G, a[i].B)

		l, e := DistanceLab(a[i], b[i])
		if e != nil {
			fmt.Println(a[i])
			os.Exit(-1)
		}
		dist += l
	}
	//fmt.Println(dist)
	return dist
}

var Searcher = 6
var Analyzer = 16

type pair struct {
	X, Y     []float64
	Filename string
}
type SortableStruct struct {
	data     []float64
	Filename string
	distance float64
}

func FindKNN(target []float64, numNeighbor int, db []TrainingData) []SortableStruct {
	best := bestResult{}
	best.best = []SortableStruct{}
	best.max = numNeighbor

	wg := sync.WaitGroup{}
	search := func(target []float64, db []TrainingData) {
		subBest := bestResult{}
		subBest.best = []SortableStruct{}
		subBest.max = numNeighbor
		defer wg.Done()
		wg2 := sync.WaitGroup{}
		c := make(chan pair, 400)
		for i := 0; i < Analyzer; i++ {
			wg2.Add(1)
			go func() {
				for job := range c {
					dist := Float64Dist(job.X, job.Y) //HclDist(db[i].Data, target)
					newItem := SortableStruct{}
					newItem.distance = dist
					newItem.data = job.X
					//fmt.Println(job.Filename)
					newItem.Filename = job.Filename //db[i].Filename
					subBest.Lock()
					subBest.Add(newItem)
					subBest.Unlock()
				}
				wg2.Done()
			}()
		}
		for i := 0; i < len(db); i++ {
			c <- pair{db[i].Signature, target, db[i].Filename}
		}
		close(c)
		wg2.Wait()
		best.Lock()
		for _, val := range subBest.best {
			best.Add(val)
		}
		best.Unlock()
	}
	for i := 0; i < Searcher; i++ {
		wg.Add(1)
		if i < Searcher-1 {
			fmt.Println(i*len(db)/Searcher, (i+1)*len(db)/Searcher)
			go search(target, db[i*len(db)/Searcher:(i+1)*len(db)/Searcher])
		} else {
			fmt.Println(i * len(db) / Searcher)
			go search(target, db[i*len(db)/Searcher:])
		}

	}
	wg.Wait()
	return best.best
}
