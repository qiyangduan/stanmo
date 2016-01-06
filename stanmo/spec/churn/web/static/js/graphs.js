jQuery.getJSON( "/get_overall_statistics", function( overallData ) { 
    d3.select("#nbr_of_inst_strong").text(overallData.number_of_model_instances); 
    d3.select("#nbr_of_feedback").text(overallData.number_of_feedbacks);  
    d3.select("#nbr_of_prediction").text("(/".concat(overallData.number_of_predictions).concat(")")); 
    d3.select("#overall_precision_strong").text(overallData.overall_precision);  
    
    /*
    function draw_by_id(selectionID, dataset){ 
        for (var col_name in dataset) {
          d3.select("body").select(selectionID).append("li");;
        }
        
        var p = d3.select("body").select(selectionID).selectAll("li")
        .data(dataset)
        .text(function(d,i){return i + ": " + d;})
     }

    // var dataset1 = [column1', 'column2', 'column 3', 'col3', 'col5'];
    // var dataset2 = ['column11', 'column12', 'column1 3', 'col13', 'co1l5','column11', 'column12', 'column1 3', 'col13', 'co1l5'];
    
    var dataset1 = overallData.input_attributes;
    var dataset2 = overallData.output_attributes;
    draw_by_id("#input_ul_1", dataset1);
    draw_by_id("#input_ul_2", dataset2);
    */
    
    });

jQuery.getJSON( "/get_prediction_history", function( lineChartData ) { 
    var ctx = document.getElementById("prediction_history_canvas").getContext("2d");
    window.myLine = new Chart(ctx).Line(lineChartData, {
            responsive: true, 
            scaleShowVerticalLines: false
        });
});

    
jQuery.getJSON( "/get_cumulative_gain_data", function( lineChartData ) { 
    drawNVD3Lines(lineChartData);
});


function drawNVD3Lines(sinCosData) {
        //sinCosData = sinAndCos();
        //sinCosData = [{"color": "#ff7f0e", "values": [{"y": 0.0, "x": 0.0}, {"y": 0.0, "x": 0.0}, {"y": 0.0, "x": 0.0}, {"y": 0.0, "x": 0.0}, {"y": 0.5, "x": 0.5}, {"y": 1.0, "x": 1.0}], "key": "Random Benchmark"}, {"color": "#2ca02c", "values": [{"y": 0.33333333333333331, "x": 0.0}, {"y": 0.5, "x": 0.0}, {"y": 0.66666666666666663, "x": 0.0}, {"y": 0.83333333333333337, "x": 0.0}, {"y": 0.83333333333333337, "x": 0.5}, {"y": 1.0, "x": 1.0}], "key": "Model ROC"}]
        nv.addGraph(function() {
        var chart = nv.models.lineChart();
        var fitScreen = false;
        var width = 600;
        var height = 450;
        var zoom = 1;
        chart.useInteractiveGuideline(true);
        chart.xAxis
            .tickFormat(d3.format(',r'));
        chart.lines.dispatch.on("elementClick", function(evt) {
            console.log(evt);
        });
        chart.yAxis
            .axisLabel('Voltage (v)')
            .tickFormat(d3.format(',.2f'));
        
        d3.select('#cumulative-gain-chart svg')
            .attr('perserveAspectRatio', 'xMinYMid')
            .attr('width', width)
            .attr('height', height)
            .datum(sinCosData);
        setChartViewBox();
        resizeChart();
        nv.utils.windowResize(resizeChart);
        d3.select('#zoomIn').on('click', zoomIn);
        d3.select('#zoomOut').on('click', zoomOut);
        function setChartViewBox() {
            var w = width * zoom,
                h = height * zoom;
            chart
                .width(w)
                .height(h);
            d3.select('#cumulative-gain-chart svg')
                .attr('viewBox', '0 0 ' + w + ' ' + h)
                .transition().duration(500)
                .call(chart);
        }
        function zoomOut() {
            zoom += .25;
            setChartViewBox();
        }
        function zoomIn() {
            if (zoom <= .5) return;
            zoom -= .25;
            setChartViewBox();
        }
        // This resize simply sets the SVG's dimensions, without a need to recall the chart code
        // Resizing because of the viewbox and perserveAspectRatio settings
        // This scales the interior of the chart unlike the above
        function resizeChart() {
            //var svg = d3.select("#cumulative-gain-chart").append("svg")     .append("g");
            var container = d3.select('#prediction-history');
            var svg = container.select('svg').append("g");

        
            if (fitScreen) {
                // resize based on container's width AND HEIGHT
                var windowSize = nv.utils.windowSize();
                svg.attr("width", windowSize.width);
                svg.attr("height", windowSize.height);
            } else {
                // resize based on container's width
                var aspect = chart.width() / chart.height();
                var targetWidth = parseInt(container.style('width'));
                svg.attr("width", targetWidth);
                svg.attr("height", Math.round(targetWidth / aspect));
            }
        }
        return chart;
    });
}

//drawNVD3Lines();

