$(document).ready(() => {
  console.log("JS Works!");

  const loaderId = "#loader";
  const inputImgId = "#img";
  const outputImgId = "#outputImg";
  const classOutputId = "#classOutput";
  const classOutputPneumoniaId = "#classOutputPneumonia";
  const classTextId = "#classText";
  const classTextPneumoniaId = "#classTextPneumonia";
  const classOutputPneumoniaWrapperId = "#classOutputPneumoniaWrapper";
  const predictButtonId = "#predictButton";

  const classOutputElement = document.querySelector(classOutputId);
  const classOutputPneumoniaElement = document.querySelector(
    classOutputPneumoniaId
  );

  // const confidenceNPId = "#confidenceNP";
  // const confidenceBVId = "#confidenceBV";

  const loaderHTML = `
    Predicting
    <div class="loader-wrapper d-inline">
      <div id="loader" class="lds-ring">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
      </div>
    </div>
    `;

  // initially hide the second output container
  $(classOutputPneumoniaWrapperId).hide();

  // helper function for POSTing an image
  const imagePostApi = (route, data, callBack) => {
    $.ajax({
      url: route,
      type: "POST",
      data: data,
      processData: false,
      contentType: false,
      success: data => callBack(data),
      error: xhr => console.error(xhr)
    });
  };

  // for sumbmitting the form
  $("form").submit(e => {
    e.preventDefault();

    // disable the button after submitting and put in the loader
    $(predictButtonId).attr("disabled", true);
    $(predictButtonId).text("Submitted");
    $(classOutputPneumoniaWrapperId).hide(); // hide second container
    $(classTextId).html(loaderHTML);

    classOutputElement.classList = "bg-info classOutput";

    // load the form data
    const formData = new FormData($("form").get(0));

    imagePostApi("/predict/NP", formData, data => {
      console.debug("NP Data Recieved:");
      console.dir(data);
      if (data.success == true) {
        $(classTextId).html(data.class);
        // $(confidenceNPId).html(data.confidence);

        switch (data.prediction) {
          case 0: // Normal Class
            $(predictButtonId).removeAttr("disabled");
            $(predictButtonId).html("Submit");
            classOutputElement.classList = "bg-success classOutput";
            break;
          case 1: // Pneumonia class
            classOutputElement.classList = "bg-custom-danger classOutput";
            $(classOutputPneumoniaWrapperId).show();
            $(classTextPneumoniaId).html(loaderHTML);

            // call second model in case of pneumonia
            imagePostApi("/predict/BV", formData, data => {
              console.debug("BV Data Recieved:");
              console.dir(data);
              if (data.success == true) {
                $(classTextPneumoniaId).html(data.class);
                // $(confidenceBVId).html(data.confidence);
                $(predictButtonId).removeAttr("disabled");
                $(predictButtonId).html("Submit");
                classOutputPneumoniaElement.classList =
                  "bg-primary classOutput";
              } else {
                console.error("Some Error Occurred at Client (BV)");
              }
            });
            break;
          default:
            classOutputElement.classList = "bg-info classOutput";
        }
      } else {
        console.error("Some Error Occurred at Client (NP)");
      }
    });
  });

  // for displaying the image preview
  $(inputImgId).on("change", e => {
    const input = e.target;
    const fileName = $(input)
      .val() //get the file name
      .replace("C:\\fakepath\\", " "); //replace the "Choose a file" label
    // display the file name
    $(input)
      .next(".custom-file-label")
      .html(fileName);

    // display the image
    if (input.files && input.files[0]) {
      let reader = new FileReader();
      reader.onload = e => {
        $(outputImgId).attr("src", e.target.result);
      };
      reader.readAsDataURL(input.files[0]);
    }
  });
});
