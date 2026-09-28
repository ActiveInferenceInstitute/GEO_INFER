# GEO-INFER-ART Data Schemas

## Overview

This document describes the data schemas used by GEO-INFER-ART for styling and visualization.

## Style Definition Schema

```yaml
# style.yaml
name: watercolor
version: "1.0"

background:
  color: "#f5f5dc"
  
layers:
  water:
    fill: "#a8d4e6"
    stroke: none
    opacity: 0.8
    
  land:
    fill: "#c5d5a9"
    stroke: "#8b956d"
    stroke_width: 1
    
  roads:
    stroke: "#4a4a4a"
    stroke_width: 2
    line_cap: round
    
  buildings:
    fill: "#d4c5b5"
    stroke: "#a09080"
    
effects:
  - type: texture
    pattern: watercolor_paper
  - type: blur
    radius: 0.5
```

## Color Palette Schema

```yaml
# palette.yaml
name: earth_tones
type: categorical

colors:
  primary: "#5d4037"
  secondary: "#8d6e63"
  accent: "#a1887f"
  
gradient:
  - position: 0
    color: "#3e2723"
  - position: 0.5
    color: "#795548"
  - position: 1.0
    color: "#d7ccc8"
```

## Map Configuration Schema

```yaml
# map_config.yaml
title: "City Map"
projection: "EPSG:3857"
extent:
  west: -122.5
  south: 37.7
  east: -122.3
  north: 37.9

output:
  format: png
  width: 1920
  height: 1080
  dpi: 300
```


