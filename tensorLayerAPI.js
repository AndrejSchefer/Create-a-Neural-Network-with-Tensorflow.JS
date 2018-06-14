//Step 1
//Require tensorflow and initialize the model
var tf = require("@tensorflow/tfjs");
const model = tf.sequential();

//Step 2
//Create a HiddenLayer
const hidden = tf.layers.dense({
    units: 4,           // Number of nodes in the hidden Layer
    inputShape: [2],    // Input Layer with 2 nodes
    activation: "sigmoid"
});

//Step 3
//Add a HiddenLayer to Model
model.add(hidden);

//Step 4
//Create a OutputLayer
const output = tf.layers.dense({
    units: 3,
    activation: "sigmoid"
});

//Step 5
//Add OutputLayer to Model
model.add(output);

//Step 6
//Generates the module
model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: tf.losses.cosineDistance
});


//Input data for the Input layer !
const xs = tf.tensor2d([
    [0.1, 0.1],
    [0.9, 0.9],
    [0.5, 0.5]
]);

//Output data
const ys = tf.tensor2d([
    [0.9, 0.9, 0.9],
    [0.1, 0.1, 0.1],
    [0.5, 0.5, 0.5]
]);

//Train the model
train().then(function () {
    console.log("training complete");
    let outputPredict = model.predict(xs);
    outputPredict.print();
});

//The train function
async function train() {
    for (var i = 0; i < 1000; i++) {
        const response = await model.fit(xs, ys, {
            shuffle: true,
            epochs: 20
        });
        // see the lostfunktion
        console.log(response.history.loss[0]);
    }
}