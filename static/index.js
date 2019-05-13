$(document).ready(() => {
  console.log("JS Works!");

  const loaderId = "#loader";
  const inputImgId = "#img";
  const outputImgId = "#outputImg";
  const classOutputId = "#classOutput";
  const classOutputWrapperId = "#classOutputWrapper";
  const predictButtonId = "#predictButton";
  const classOutputElement = document.querySelector(classOutputId);
  const classOutputWrapperElement = document.querySelector(
    classOutputWrapperId
  );

  // for sumbmitting the form
  $("form").submit(e => {
    e.preventDefault();

    // disable the button after submitting and put in the loader
    $(predictButtonId).attr("disabled", true);
    $(predictButtonId).text("Submitted");

    $(classOutputId).html(`
    Predicting
    <div class="loader-wrapper d-inline">
      <div id="loader" class="lds-ring">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
      </div>
    </div>
    `);

    // load the form data
    const formData = new FormData($("form").get(0));

    $.ajax({
      url: "/predict",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      beforeSend: () => {
        console.log("In beforeSend()");
        classOutputElement.classList = "bg-info";
      },
      success: data => {
        console.debug("Data Recieved:");
        console.dir(data);
        if (data.success == true) {
          $(classOutputId).html(data.class);
          $(predictButtonId).removeAttr("disabled");
          $(predictButtonId).html("Submit");

          switch (data.prediction) {
            case 0: // Normal Class
              classOutputElement.classList = "bg-success";
              break;
            case 1: // Bacterial class
            case 2: // Viral class
              classOutputElement.classList = "bg-custom-danger";
              break;
            default:
              classOutputElement.classList = "bg-info";
          }

        } else {
          console.error("Some Error Occurred at Client");
        }
      },
      error: xhr => {
        console.error(xhr);
      }
    });
  });

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
