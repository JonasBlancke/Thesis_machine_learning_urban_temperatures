
var colorizedVis = {
  min: 0,
  max: 1,
  palette: [
    'ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901',
    '66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01',
    '012e01', '011d01', '011301'
  ],
};

// Define the start and end dates for each month from 2008 to 2017.
var months = [];
for (var year = 2008; year <= 2017; year++) {
  for (var month = 1; month <= 12; month++) {
    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month').advance(-1, 'day');
    var name = year + "_" + month;
    months.push({ name: name, start: start, end: end });
  }
}

// Loop through the months and export colorized NDVI images.
for (var i = 0; i < months.length; i++) {
  var month = months[i];
  var dataset = ee.ImageCollection('MODIS/061/MOD13Q1')
    .filterDate(month.start, month.end)
    .filterBounds(Europe);
  var colorized = dataset.select('NDVI').median();
  
  // Clip the colorized NDVI image to the Ghent region.
  var clippedColorized = colorized.clip(Europe);
  
  // Visualize and add the layer to the map.
  Map.addLayer(clippedColorized, colorizedVis, month.name + " Colorized NDVI");
  
  // Define export options and export the clipped image.
  var exportOptions = {
    image: clippedColorized,
    description: "Colorized_NDVI_" + month.name,
    scale: 250,  // Adjust the scale as needed.
    fileFormat: "GeoTIFF"
  };
  Export.image.toDrive(exportOptions);
}

print("Export tasks started. Check the 'Tasks' tab for progress.");
