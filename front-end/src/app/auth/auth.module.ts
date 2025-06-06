import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { TranslateModule } from '@ngx-translate/core';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';

import { I18nModule } from '@app/i18n';
import { AuthRoutingModule } from './auth-routing.module';
import { LoginComponent } from './login.component';
import { UserCreateComponent } from '../user-create/user-create.component';

@NgModule({
  imports: [CommonModule, ReactiveFormsModule, TranslateModule, FormsModule, NgbModule, I18nModule, AuthRoutingModule],
  declarations: [LoginComponent, UserCreateComponent],
})
export class AuthModule {}
