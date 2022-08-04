// Plotly.d3.csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv", function(err, rows){

//   function unpack(rows, key) {
//   return rows.map(function(row) { return row[key]; });
// }


// data = [
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-1"),
//       theta: unpack(rows, "Time"),
//       name: 'SystemTrack1',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-2"),
//        theta: unpack(rows, "Time"),
//       name: 'SystemTrack2',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-3"),
//       theta: unpack(rows, "Time"),
//       name: 'SystemTrack3',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "Time"),
//       theta: unpack(rows, "Time"),
//       name: 'SystemTrack4',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-5"),
//        theta: unpack(rows, "Time"),
//       name: 'SystemTrack5',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-6"),
//       theta: unpack(rows, "Time"),
//       name: 'SystemTrack6',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-7"),
//        theta: unpack(rows, "Time"),
//       name: 'SystemTrack7',
//       line: {
//         width: 10,
//       }
//     },
//     {
//       type: 'scatterpolar',
//       r: unpack(rows, "SystemTrack-8"),
//       theta: unpack(rows, "Time"),
//       name: 'SystemTrack8',
//       line: {
//         width: 10,
//       }
//     }
    
//     ]
    
//     layout = {
//       polar: {
//         radialaxis: {
//           visible: true,
//           range: [0, 7]
//         },
//       },
//       showlegend: true
//     }
//     Plotly.newPlot('myDiv', data, layout);

//     // return {
//     //     data : data,
//     //     layout : layout,
//     //     config : {
//     //     displayModeBar: false
//     //     }
//     //     }
    
// })

d3.csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv", function(err, rows){

  function unpack(rows, key) {
  return rows.map(function(row) { return row[key]; });
}


var trace1 = {
  type: "scatter",
  mode: "lines",
  name: 'AAPL High',
  x: unpack(rows, 'Date'),
  y: unpack(rows, 'AAPL.High'),
  line: {color: '#17BECF'}
}

var trace2 = {
  type: "scatter",
  mode: "lines",
  name: 'AAPL Low',
  x: unpack(rows, 'Date'),
  y: unpack(rows, 'AAPL.Low'),
  line: {color: '#7F7F7F'}
}

var data = [trace1,trace2];

var layout = {
  title: 'Basic Time Series',
};

Plotly.newPlot('myDiv', data, layout);
})
