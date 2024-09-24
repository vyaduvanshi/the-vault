document.querySelectorAll('input[name="payment_method"]').forEach((elem) => {
    elem.addEventListener("change", function(event) {
      var upiDetails = document.getElementById("upi-details");
      var cardDetails = document.getElementById("card-details");
      if (event.target.value === "upi") {
        upiDetails.style.display = "block";
        cardDetails.style.display = "none";
      } else {
        upiDetails.style.display = "none";
        cardDetails.style.display = "block";
      }
    });
});